# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import numpy as np
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from enum import Enum
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from unittest.mock import patch


def cleanup_train(rank):
    if rank >= 0:
        dist.destroy_process_group()


def create_folder(path, print_if_create=False):
    if not os.path.exists(path):
        os.makedirs(path)
        if print_if_create:
            print(f'created folder: {path}')


@patch('json.encoder.c_make_encoder', None)
def dump_json(obj, fid, float_digits=-1, **kwargs):
    of = json.encoder._make_iterencode

    def inner(*args, **kwargs):
        args = list(args)
        # fifth argument is float formatter which we may replace
        if float_digits is not None and float_digits >= 0:
            args[4] = lambda o: format(o, f'.{int(float_digits)}f')
        return of(*args, **kwargs)

    with patch('json.encoder._make_iterencode', wraps=inner):
        json.dump(obj, fid, **kwargs)


def generate_log_json(frame_num, frame_pixel_num, test_time, frame_types, bits, psnrs, ssims,
                      verbose=False, avg_encoding_time=None, avg_decoding_time=None):
    include_yuv = len(psnrs[0]) > 1
    assert not include_yuv or (len(psnrs[0]) == 4 and len(ssims[0]) == 4)

    # Accumulate stats for i-frames (frame_type==0) and p-frames (frame_type!=0)
    metrics = ['psnr', 'ssim'] + (['psnr_y', 'psnr_u', 'psnr_v', 'ssim_y', 'ssim_u', 'ssim_v']
                                  if include_yuv else [])
    metric_idx = {'psnr': 0, 'ssim': 0, 'psnr_y': 1, 'psnr_u': 2, 'psnr_v': 3,
                  'ssim_y': 1, 'ssim_u': 2, 'ssim_v': 3}
    i_sum = {m: 0 for m in metrics}
    p_sum = {m: 0 for m in metrics}
    i_sum['bits'], p_sum['bits'] = 0, 0
    i_num, p_num = 0, 0

    for idx in range(frame_num):
        is_i_frame = frame_types[idx] == 0
        target = i_sum if is_i_frame else p_sum
        target['bits'] += bits[idx]
        for m in metrics:
            src = psnrs if 'psnr' in m else ssims
            target[m] += src[idx][metric_idx[m]]
        if is_i_frame:
            i_num += 1
        else:
            p_num += 1

    log_result = {
        'frame_pixel_num': frame_pixel_num,
        'i_frame_num': i_num,
        'p_frame_num': p_num,
    }
    for prefix, num, sums in [('i', i_num, i_sum), ('p', p_num, p_sum)]:
        log_result[f'ave_{prefix}_frame_bpp'] = (
            sums['bits'] / num / frame_pixel_num if num > 0 else 0)
        log_result[f'ave_{prefix}_frame_psnr'] = sums['psnr'] / num if num > 0 else 0
        log_result[f'ave_{prefix}_frame_msssim'] = sums['ssim'] / num if num > 0 else 0
        if include_yuv:
            for suffix in ['y', 'u', 'v']:
                log_result[f'ave_{prefix}_frame_psnr_{suffix}'] = (
                    sums[f'psnr_{suffix}'] / num if num > 0 else 0)
                log_result[f'ave_{prefix}_frame_msssim_{suffix}'] = (
                    sums[f'ssim_{suffix}'] / num if num > 0 else 0)

    if verbose:
        log_result['frame_bpp'] = list(np.array(bits) / frame_pixel_num)
        log_result['frame_psnr'] = [v[0] for v in psnrs]
        log_result['frame_msssim'] = [v[0] for v in ssims]
        log_result['frame_type'] = frame_types
        if include_yuv:
            for suffix, idx in [('y', 1), ('u', 2), ('v', 3)]:
                log_result[f'frame_psnr_{suffix}'] = [v[idx] for v in psnrs]
                log_result[f'frame_msssim_{suffix}'] = [v[idx] for v in ssims]

    log_result['test_time'] = test_time

    total_bits = i_sum['bits'] + p_sum['bits']
    log_result['ave_all_frame_bpp'] = total_bits / (frame_num * frame_pixel_num)
    log_result['ave_all_frame_psnr'] = (i_sum['psnr'] + p_sum['psnr']) / frame_num
    log_result['ave_all_frame_msssim'] = (i_sum['ssim'] + p_sum['ssim']) / frame_num
    if avg_encoding_time is not None and avg_decoding_time is not None:
        log_result['avg_frame_encoding_time'] = avg_encoding_time
        log_result['avg_frame_decoding_time'] = avg_decoding_time
    if include_yuv:
        for suffix in ['y', 'u', 'v']:
            log_result[f'ave_all_frame_psnr_{suffix}'] = (
                i_sum[f'psnr_{suffix}'] + p_sum[f'psnr_{suffix}']) / frame_num
            log_result[f'ave_all_frame_msssim_{suffix}'] = (
                i_sum[f'ssim_{suffix}'] + p_sum[f'ssim_{suffix}']) / frame_num

    return log_result


def generate_str(x):
    return '  '.join(f'{a.item():.5f}' for a in x) + '  '


def get_current_device(rank):
    if rank <= 0:
        print(f'cuda device count: {torch.cuda.device_count()}')

    if rank >= 0:
        device = f'cuda:{rank}'
    elif torch.cuda.device_count() > 0:
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(f'rank: {rank}, current device: {device}')
    return device


def get_dataloader(dataset, rank, world_size, batch_size, num_workers):
    if rank >= 0:
        train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        assert batch_size % world_size == 0
        arg_batch_size = batch_size // world_size
    else:
        train_sampler = RandomSampler(dataset)
        arg_batch_size = batch_size
    return DataLoader(
        dataset,
        batch_size=arg_batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
        prefetch_factor=2,
    )


def get_latest_status_path(dir_cur):
    files = os.listdir(dir_cur)
    all_status_files = [os.path.join(dir_cur, f) for f in files if 'status_epo' in f]
    all_status_files.sort(key=os.path.getmtime)
    if len(all_status_files) > 2:
        return all_status_files[-2:]
    return all_status_files


def loss_func(rd, lambdas):
    costs = lambdas * rd['mse'] + rd['bpp']
    return {
        'losses': costs,
        'loss': torch.mean(costs),
    }


def get_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=True)
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    if 'net' in ckpt:
        ckpt = ckpt['net']
    consume_prefix_in_state_dict_if_present(ckpt, prefix='module.')
    return ckpt


def get_training_lambdas(lambdas, qp_num):
    all_lambdas = np.linspace(np.log(lambdas[0]), np.log(lambdas[1]), qp_num)
    all_lambdas = np.exp(all_lambdas)
    return all_lambdas


def init_train(rank, save_dir):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    world_size = 1
    if rank >= 0:
        torch.cuda.set_device(rank)
        world_size = torch.cuda.device_count()
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=world_size, rank=rank)

    if rank <= 0:
        create_folder(save_dir)

    return world_size


def load_existing_weights(save_dir, net, rank, pretrain_path=None):
    begin_epoch = 0
    opt_status = None
    ckpt_loaded = False
    existing_status_path = get_latest_status_path(save_dir)
    for status_path in reversed(existing_status_path):
        try:
            status = torch.load(status_path, map_location=torch.device('cpu'),
                                weights_only=True)
            opt_status = status['opt']
            begin_epoch = status['epoch'] + 1
            ckpt_loaded = True
            if rank <= 0:
                net_status = status['net']
                consume_prefix_in_state_dict_if_present(net_status, prefix='module.')
                net.load_state_dict(net_status)
                print(f'load status from {status_path}')
                print(f'begin epoch {begin_epoch}')
            break
        except Exception:
            continue

    if not ckpt_loaded and pretrain_path is not None:
        if rank <= 0:
            net_state_dict = get_state_dict(pretrain_path)
            net.load_state_dict(net_state_dict)
            print(f'load pretrained weights from {pretrain_path}')

    return begin_epoch, opt_status


def save_ckpt(save_dir, net):
    ckpt_path = os.path.join(save_dir, 'ckpt.pth.tar')
    net = net.module if isinstance(net, DDP) else net
    torch.save({'state_dict': net.state_dict()}, ckpt_path)
    print(f'save final checkpoint to {ckpt_path}')


def save_status(save_dir, net, opt, epoch):
    curr_path = os.path.join(save_dir, f'status_epo{epoch}.pth.tar')
    net = net.module if isinstance(net, DDP) else net
    save_dict = {
        'epoch': epoch,
        'net': net.state_dict(),
        'opt': opt.state_dict(),
    }
    torch.save(save_dict, curr_path)
    print(f'save model epoch {epoch}')

    for f in os.listdir(save_dir):
        full_path = os.path.join(save_dir, f)
        if 'status_epo' in f and full_path != curr_path:
            if os.path.exists(full_path):
                os.remove(full_path)


def set_torch_env():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    np.random.seed(seed=0)
    torch.utils.deterministic.fill_uninitialized_memory = False


def start_train(train_fun, args):
    if torch.cuda.device_count() > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(12355)
        world_size = torch.cuda.device_count()
        mp.spawn(train_fun, nprocs=world_size, args=(args,), join=True)
    else:
        train_fun(-1, args)


def str2bool(v):
    return str(v).lower() in ['yes', 'y', 'true', 't', '1']


def wrap_ddp(net, rank, find_unused_parameters=False):
    if rank >= 0:
        net = DDP(net, device_ids=[rank], find_unused_parameters=find_unused_parameters)
    return net


class ModelStructure(Enum):
    HTL = 'htl'
    HTS = 'hts'
    LD = 'ld'
