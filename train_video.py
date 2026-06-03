# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import math
import sys
import time
import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_

from src.datasets.video_dataset import VideoFolder
from src.models.image_model import DMCI
from src.utils.common import ModelStructure, save_status, start_train, save_ckpt, \
    get_current_device, get_training_lambdas, get_dataloader, init_train,  cleanup_train, \
    load_existing_weights, get_state_dict, wrap_ddp


def get_training_strategy(training_scheduling, frame_delay, model_structure):
    # epoch is for referencing purpose
    # lr is the learning rate for the current epoch
    if model_structure == ModelStructure.LD:
        if training_scheduling == 'stage0':
            training_strategy = \
                [[0,   1e-4, 2, False, 256, 256]] * 5 + \
                [[5,   1e-4, 3, False, 256, 256]] * 5 + \
                [[10,  1e-4, 6, False, 256, 256]] * 45 + \
                [[55,  1e-4, 6, False, 256, 256]]  # noqa: E501 E221
                # epo, lr, frame_num, cascade, patch_size  # noqa: E116 E501
        elif training_scheduling == 'stage1':
            training_strategy = \
                [[0,   5e-5,  8, True,  256, 256]] * 5 + \
                [[5,   5e-5, 16, True,  256, 256]] * 5 + \
                [[10,  5e-5, 24, True,  256, 256]] * 5 + \
                [[15,  5e-5, 32, True,  256, 256]] * 15 + \
                [[30,  5e-6, 32, True,  256, 256]] * 7 + \
                [[37,  5e-6, 32, True,  256, 256]]  # noqa: E501 E221
                # epo, lr, frame_num, cascade, patch_size  # noqa: E116 E501
        elif training_scheduling == 'stage2':
            training_strategy = \
                [[0,   5e-5, 33, True,  512, 512]] * 14 + \
                [[14,  5e-6, 33, True,  512, 512]] * 4 + \
                [[18,  2e-5, 49, True,  512, 512]] * 7 + \
                [[25,  2e-6, 49, True,  512, 512]] * 2 + \
                [[27,  5e-6, 65, True,  512, 512]] * 7 + \
                [[34,  2e-6, 65, True,  512, 512]] * 6 + \
                [[40,  2e-6, 65, True,  512, 512]]  # noqa: E501 E221
                # epo, lr, frame_num, cascade, patch_size  # noqa: E116 E501
        elif training_scheduling == 'stage3':
            training_strategy = \
                [[0,   2e-6, 97,  True,  512, 512]] * 2 + \
                [[2,   5e-7, 129, True,  512, 512]] * 2 + \
                [[4,   5e-7, 129, True,  512, 512]]  # noqa: E501 E221
                # epo, lr, frame_num, cascade, patch_size  # noqa: E116 E501
        else:
            assert False

        return training_strategy

    if training_scheduling == 'stage0':
        training_strategy = \
            [[0,   1e-4, 1 + 1 * frame_delay, False, 256, 256]] * 5 + \
            [[5,   1e-4, 1 + 2 * frame_delay, False, 256, 256]] * 5 + \
            [[10,  1e-4, 1 + 4 * frame_delay, False, 256, 256]] * 35 + \
            [[45,  1e-4, 1 + 4 * frame_delay, False, 256, 256]]  # noqa: E501 E221
            # epo, lr, frame_num, cascade, patch_size  # noqa: E116 E501
    elif training_scheduling == 'stage1':
        training_strategy = \
            [[0,   5e-5, 17, True,  256, 256]] * 2 + \
            [[2,   5e-5, 25, True,  256, 256]] * 1 + \
            [[3,   5e-5, 33, True,  256, 256]] * 3 + \
            [[6,   5e-6, 33, True,  256, 256]] * 4 + \
            [[10,  5e-6, 33, True,  256, 256]]  # noqa: E501 E221
            # epo, lr, frame_num, cascade, patch_size  # noqa: E116 E501
    elif training_scheduling == 'stage2':
        training_strategy = \
            [[0,   5e-5, 33, True,  512, 512]] * 10 + \
            [[10,  5e-5, 49, True,  512, 512]] * 10 + \
            [[20,  1e-5, 65, True,  512, 512]] * 12 + \
            [[32,  2e-6, 65, True,  512, 512]] * 8 + \
            [[40,  2e-6, 65, True,  512, 512]]  # noqa: E501 E221
            # epo, lr, frame_num, cascade, patch_size  # noqa: E116 E501
    elif training_scheduling == 'stage3':
        training_strategy = \
            [[0,   1e-5, 97,  True,  512, 512]] * 2 + \
            [[2,   2e-6, 129, True,  512, 512]] * 2 + \
            [[4,   2e-6, 129, True,  512, 512]]  # noqa: E501 E221
            # epo, lr, frame_num, cascade, patch_size  # noqa: E116 E501
    else:
        assert False

    return training_strategy


def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('-e', '--epochs', default=100, type=int)
    parser.add_argument('--lambdas', type=float, nargs='+', required=True)
    parser.add_argument('--model_path_i', type=str, required=True)
    parser.add_argument('--model_structure', type=str, default='ld', choices=['htl', 'hts', 'ld'])
    parser.add_argument('-n', '--num_workers', type=int, default=4,
                        help='Dataloaders worker per trainer')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save models')
    parser.add_argument('--train_dataset', type=str, required=True)
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--training_scheduling', type=str, default='stage0',
                        choices=['stage0', 'stage1', 'stage2', 'stage3'],
                        help='How to schedule the training strategy')

    args = parser.parse_args(argv)
    return args


def train_one_epoch(p_net, i_net, args, dataloader, optimizer, epoch, rank,
                    frame_delay, model_structure):
    p_net_module = p_net.module if isinstance(p_net, DDP) else p_net
    training_strategy = get_training_strategy(args.training_scheduling, frame_delay,
                                              model_structure)
    p_net.train()
    device = next(p_net.parameters()).device

    idx = min(len(training_strategy) - 1, epoch)
    _, lr, train_seq_length, is_cascaded, patch_width, patch_height = training_strategy[idx]

    for g in optimizer.param_groups:
        g['lr'] = lr

    world_size = dist.get_world_size() if rank >= 0 else 1
    use_ckpt = is_cascaded and patch_width > 256
    p_net_module.set_use_ckpt(use_ckpt)

    dataloader.dataset.set_frame_num(train_seq_length)
    dataloader.dataset.set_patch_size(patch_width, patch_height)

    t0 = time.time()
    for i, batch in enumerate(dataloader):
        batch = [t.to(device) for t in batch]
        batch_size = batch[0].size(0)
        frame_nums = len(batch) - 2
        frame_stop = 2 if is_cascaded else frame_nums
        qp = batch[-2]
        curr_lambdas = batch[-1]

        print_loss_info = rank <= 0 and i % (100 if train_seq_length >= 8 else 200) == 0

        ref_frame = batch[0]
        if train_seq_length > 1 + 1 * frame_delay or is_cascaded:
            with torch.inference_mode():
                ref_frame = i_net.forward_one_frame(ref_frame, qp, recon_only=True)

        p_net_module.clear_dpb()
        p_net_module.add_ref_feature_from_frame(ref_frame, apply_feature_adaptor=False)

        for frame_idx in range(1, frame_stop):
            if is_cascaded:
                cur_frame = batch[1:-2]
            else:
                cur_frame = batch[frame_idx]
            loss, info = p_net(cur_frame, qp, lambdas=curr_lambdas, get_loss_info=print_loss_info,
                               curr_poc=frame_idx)

            optimizer.zero_grad()
            loss.backward()

            total_norm = clip_grad_norm_(p_net.parameters(), max_norm=0.2,
                                         error_if_nonfinite=False).item()
            if math.isnan(total_norm) or math.isinf(total_norm):
                if rank <= 0:
                    print('non-finite norm, skip this batch')
                continue

            optimizer.step()

        if print_loss_info:
            numer = i * batch_size * world_size
            denom = len(dataloader.dataset)
            t1 = time.time()
            print(
                f'Time: {t1-t0:.3f} seconds, Train epoch {epoch}:'
                f' [{numer} / {denom} ({100. * numer / denom:.0f}%)]'
                f' MSE: {info['mse']} | Bpp y: {info['bpp_y']} | Bpp z: {info['bpp_z']} |'
                f' Losses: {info['losses']} | lr: {optimizer.param_groups[0]['lr']:.1e}'
            )
            t0 = t1


def train(rank, args):
    world_size = init_train(rank, args.save_dir)
    model_structure = ModelStructure(args.model_structure)
    if model_structure == ModelStructure.LD:
        from src.models.video_model_ld import DMC, g_frame_delay as frame_delay
    else:
        from src.models.video_model_ht import DMC, g_frame_delay as frame_delay
    train_dataset = VideoFolder(args.train_dataset, 256, 256, DMCI.qp_num(),
                                get_training_lambdas(args.lambdas, DMCI.qp_num()),
                                group_of_pictures=frame_delay)

    i_state_dict = get_state_dict(args.model_path_i)
    i_net = DMCI()
    i_net.load_state_dict(i_state_dict)
    i_net = i_net.eval()

    if model_structure == ModelStructure.LD:
        p_net = DMC()
    else:
        p_net = DMC(model_structure=model_structure)
    begin_epoch, opt_status = load_existing_weights(
        args.save_dir, p_net, rank, args.pretrain_path)

    device = get_current_device(rank)
    if rank >= 0:
        p_net = p_net.cuda(rank)
        i_net = i_net.cuda(rank)
    else:
        p_net = p_net.to(device)
        i_net = i_net.to(device)
    p_net = wrap_ddp(p_net, rank, find_unused_parameters=True)

    optimizer = torch.optim.AdamW(p_net.parameters(), lr=1e-4)
    if opt_status is not None:
        optimizer.load_state_dict(opt_status)

    dataloader = get_dataloader(train_dataset, rank, world_size, args.batch_size,
                                args.num_workers)
    for epoch in range(begin_epoch, args.epochs):
        if rank >= 0:
            dataloader.sampler.set_epoch(epoch)
        train_one_epoch(p_net, i_net, args, dataloader, optimizer, epoch, rank,
                        frame_delay, model_structure)

        if rank <= 0:
            save_status(args.save_dir, p_net, optimizer, epoch)

    if rank <= 0:
        save_ckpt(args.save_dir, p_net)
    cleanup_train(rank)


def main(argv):
    args = parse_args(argv)
    start_train(train, args)


if __name__ == '__main__':
    main(sys.argv[1:])
