# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import concurrent.futures
import json
import multiprocessing
import time
import warnings

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from src.models.video_net_dmc import DMC
from src.models.priors import model_architectures as architectures
from src.models.utils import get_padding_size
from tqdm import tqdm
from pytorch_msssim import ms_ssim


warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="Example testing script")

    parser.add_argument('--i_frame_model_name', type=str, default="IntraNoAR")
    parser.add_argument('--i_frame_model_path', type=str, nargs="+")
    parser.add_argument("--force_intra", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--force_frame_num", type=int, default=-1)
    parser.add_argument("--force_intra_period", type=int, default=-1)
    parser.add_argument('--model_path',  type=str, nargs="+")
    parser.add_argument('--test_config', type=str, required=True)
    parser.add_argument("--worker", "-w", type=int, default=1, help="worker number")
    parser.add_argument("--cuda", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--cuda_device", default=None,
                        help="the cuda device used, e.g., 0; 0,1; 1,2,3; etc.")
    parser.add_argument('--write_stream', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--stream_path', type=str, default="out_bin")
    parser.add_argument('--save_decoded_frame', type=str2bool, default=False)
    parser.add_argument('--decoded_frame_path', type=str, default='decoded_frames')
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--verbose', type=int, default=0)

    args = parser.parse_args()
    return args


def read_image_to_torch(path):
    input_image = Image.open(path).convert('RGB')
    input_image = np.asarray(input_image).astype('float64').transpose(2, 0, 1)
    input_image = torch.from_numpy(input_image).type(torch.FloatTensor)
    input_image = input_image.unsqueeze(0)/255
    return input_image


def save_torch_image(img, save_path):
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)


def PSNR(input1, input2):
    mse = torch.mean((input1 - input2) ** 2)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr.item()


def run_test(video_net, i_frame_net, args_dict, device):
    frame_num = args_dict['frame_num']
    gop_size = args_dict['gop_size']
    write_stream = 'write_stream' in args_dict and args_dict['write_stream']
    save_decoded_frame = 'save_decoded_frame' in args_dict and args_dict['save_decoded_frame']
    verbose = args_dict['verbose'] if 'verbose' in args_dict else 0

    pngs = os.listdir(args_dict['img_path'])
    if 'im1.png' in pngs:
        padding = 1
    elif 'im00001.png' in pngs:
        padding = 5
    else:
        raise ValueError('unknown image naming convention; please specify')

    frame_types = []
    psnrs = []
    msssims = []
    bits = []
    bits_mv_y = []
    bits_mv_z = []
    bits_y = []
    bits_z = []
    frame_pixel_num = 0

    start_time = time.time()
    p_frame_number = 0
    overall_p_decoding_time = 0
    with torch.no_grad():
        for frame_idx in range(frame_num):
            frame_start_time = time.time()
            x = read_image_to_torch(os.path.join(
                args_dict['img_path'],
                f"im{str(frame_idx+1).zfill(padding)}.png"
            ))
            x = x.to(device)
            pic_height = x.shape[2]
            pic_width = x.shape[3]

            if frame_pixel_num == 0:
                frame_pixel_num = x.shape[2] * x.shape[3]
            else:
                assert frame_pixel_num == x.shape[2] * x.shape[3]

            # pad if necessary
            height, width = x.size(2), x.size(3)
            padding_l, padding_r, padding_t, padding_b = get_padding_size(height, width)
            x_padded = torch.nn.functional.pad(
                x,
                (padding_l, padding_r, padding_t, padding_b),
                mode="constant",
                value=0,
            )

            bin_path = os.path.join(args_dict['bin_folder'], f"{frame_idx}.bin") \
                if write_stream else None

            if frame_idx % gop_size == 0:
                result = i_frame_net.encode_decode(x_padded, bin_path,
                                                   pic_height=pic_height, pic_width=pic_width)
                ref_frame = result["x_hat"]
                ref_feature = None
                frame_types.append(0)
                bits.append(result["bit"])
                bits_mv_y.append(0)
                bits_mv_z.append(0)
                bits_y.append(0)
                bits_z.append(0)
            else:
                result = video_net.encode_decode(x_padded, ref_frame, ref_feature, bin_path,
                                                 pic_height=pic_height, pic_width=pic_width)
                ref_frame = result['x_hat']
                ref_feature = result['feature']
                frame_types.append(1)
                bits.append(result['bit'])
                bits_mv_y.append(result['bit_mv_y'])
                bits_mv_z.append(result['bit_mv_z'])
                bits_y.append(result['bit_y'])
                bits_z.append(result['bit_z'])
                p_frame_number += 1
                overall_p_decoding_time += result['decoding_time']

            ref_frame = ref_frame.clamp_(0, 1)
            x_hat = F.pad(ref_frame, (-padding_l, -padding_r, -padding_t, -padding_b))
            psnr = PSNR(x_hat, x)
            msssim = ms_ssim(x_hat, x, data_range=1).item()
            psnrs.append(psnr)
            msssims.append(msssim)
            frame_end_time = time.time()

            if verbose >= 2:
                print(f"frame {frame_idx}, {frame_end_time - frame_start_time:.3f} seconds,",
                      f"bits: {bits[-1]:.3f}, PSNR: {psnrs[-1]:.4f}, MS-SSIM: {msssims[-1]:.4f} ")

            if save_decoded_frame:
                save_path = os.path.join(args_dict['decoded_frame_folder'], f'{frame_idx}.png')
                save_torch_image(x_hat, save_path)

    test_time = time.time() - start_time
    if verbose >= 1 and p_frame_number > 0:
        print(f"decoding {p_frame_number} P frames, "
              f"average {overall_p_decoding_time/p_frame_number * 1000:.0f} ms.")

    cur_ave_i_frame_bit = 0
    cur_ave_i_frame_psnr = 0
    cur_ave_i_frame_msssim = 0
    cur_ave_p_frame_bit = 0
    cur_ave_p_frame_bit_mv_y = 0
    cur_ave_p_frame_bit_mv_z = 0
    cur_ave_p_frame_bit_y = 0
    cur_ave_p_frame_bit_z = 0
    cur_ave_p_frame_psnr = 0
    cur_ave_p_frame_msssim = 0
    cur_i_frame_num = 0
    cur_p_frame_num = 0
    for idx in range(frame_num):
        if frame_types[idx] == 0:
            cur_ave_i_frame_bit += bits[idx]
            cur_ave_i_frame_psnr += psnrs[idx]
            cur_ave_i_frame_msssim += msssims[idx]
            cur_i_frame_num += 1
        else:
            cur_ave_p_frame_bit += bits[idx]
            cur_ave_p_frame_bit_mv_y += bits_mv_y[idx]
            cur_ave_p_frame_bit_mv_z += bits_mv_z[idx]
            cur_ave_p_frame_bit_y += bits_y[idx]
            cur_ave_p_frame_bit_z += bits_z[idx]
            cur_ave_p_frame_psnr += psnrs[idx]
            cur_ave_p_frame_msssim += msssims[idx]
            cur_p_frame_num += 1

    log_result = {}
    log_result['frame_pixel_num'] = frame_pixel_num
    log_result['i_frame_num'] = cur_i_frame_num
    log_result['p_frame_num'] = cur_p_frame_num
    log_result['ave_i_frame_bpp'] = cur_ave_i_frame_bit / cur_i_frame_num / frame_pixel_num
    log_result['ave_i_frame_psnr'] = cur_ave_i_frame_psnr / cur_i_frame_num
    log_result['ave_i_frame_msssim'] = cur_ave_i_frame_msssim / cur_i_frame_num
    log_result['frame_bpp'] = list(np.array(bits) / frame_pixel_num)
    log_result['frame_bpp_mv_y'] = list(np.array(bits_mv_y) / frame_pixel_num)
    log_result['frame_bpp_mv_z'] = list(np.array(bits_mv_z) / frame_pixel_num)
    log_result['frame_bpp_y'] = list(np.array(bits_y) / frame_pixel_num)
    log_result['frame_bpp_z'] = list(np.array(bits_z) / frame_pixel_num)
    log_result['frame_psnr'] = psnrs
    log_result['frame_msssim'] = msssims
    log_result['frame_type'] = frame_types
    log_result['test_time'] = test_time
    if cur_p_frame_num > 0:
        total_p_pixel_num = cur_p_frame_num * frame_pixel_num
        log_result['ave_p_frame_bpp'] = cur_ave_p_frame_bit / total_p_pixel_num
        log_result['ave_p_frame_bpp_mv_y'] = cur_ave_p_frame_bit_mv_y / total_p_pixel_num
        log_result['ave_p_frame_bpp_mv_z'] = cur_ave_p_frame_bit_mv_z / total_p_pixel_num
        log_result['ave_p_frame_bpp_y'] = cur_ave_p_frame_bit_y / total_p_pixel_num
        log_result['ave_p_frame_bpp_z'] = cur_ave_p_frame_bit_z / total_p_pixel_num
        log_result['ave_p_frame_psnr'] = cur_ave_p_frame_psnr / cur_p_frame_num
        log_result['ave_p_frame_msssim'] = cur_ave_p_frame_msssim / cur_p_frame_num
    else:
        log_result['ave_p_frame_bpp'] = 0
        log_result['ave_p_frame_psnr'] = 0
        log_result['ave_p_frame_msssim'] = 0
        log_result['ave_p_frame_bpp_mv_y'] = 0
        log_result['ave_p_frame_bpp_mv_z'] = 0
        log_result['ave_p_frame_bpp_y'] = 0
        log_result['ave_p_frame_bpp_z'] = 0
    log_result['ave_all_frame_bpp'] = (cur_ave_i_frame_bit + cur_ave_p_frame_bit) / \
        (frame_num * frame_pixel_num)
    log_result['ave_all_frame_psnr'] = (cur_ave_i_frame_psnr + cur_ave_p_frame_psnr) / frame_num
    log_result['ave_all_frame_msssim'] = (cur_ave_i_frame_msssim + cur_ave_p_frame_msssim) / \
        frame_num
    return log_result


def encode_one(args_dict, device):
    i_frame_load_checkpoint = torch.load(args_dict['i_frame_model_path'],
                                         map_location=torch.device('cpu'))
    if "state_dict" in i_frame_load_checkpoint:
        i_frame_load_checkpoint = i_frame_load_checkpoint['state_dict']
    i_frame_net = architectures[args_dict['i_frame_model_name']].from_state_dict(
        i_frame_load_checkpoint).eval()
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()

    if args_dict['force_intra']:
        video_net = None
    else:
        video_net = DMC()
        load_checkpoint = torch.load(args_dict['model_path'], map_location=torch.device('cpu'))
        if "state_dict" in load_checkpoint:
            load_checkpoint = load_checkpoint['state_dict']
        video_net.load_dict(load_checkpoint)
        video_net = video_net.to(device)
        video_net.eval()

    if args_dict['write_stream']:
        if video_net is not None:
            video_net.update(force=True)
        i_frame_net.update(force=True)

    sub_dir_name = args_dict['video_path']

    gop_size = args_dict['gop']
    frame_num = args_dict['frame_num']

    # to do, need to add something related to the models or model index to the bin_folder
    bin_folder = os.path.join(args_dict['stream_path'], sub_dir_name, str(args_dict['model_idx']))
    if args_dict['write_stream'] and not os.path.exists(bin_folder):
        os.makedirs(bin_folder)

    if args_dict['save_decoded_frame']:
        decoded_frame_folder = os.path.join(args_dict['decoded_frame_path'], sub_dir_name,
                                            str(args_dict['model_idx']))
        os.makedirs(decoded_frame_folder, exist_ok=True)
    else:
        decoded_frame_folder = None

    args_dict['img_path'] = os.path.join(args_dict['dataset_path'], sub_dir_name)
    args_dict['gop_size'] = gop_size
    args_dict['frame_num'] = frame_num
    args_dict['bin_folder'] = bin_folder
    args_dict['decoded_frame_folder'] = decoded_frame_folder

    result = run_test(video_net, i_frame_net, args_dict, device=device)

    result['name'] = f"{os.path.basename(args_dict['model_path'])}_{sub_dir_name}"
    result['ds_name'] = args_dict['ds_name']
    result['video_path'] = args_dict['video_path']

    return result


def worker(use_cuda, args):
    torch.backends.cudnn.benchmark = False
    if 'use_deterministic_algorithms' in dir(torch):
        torch.use_deterministic_algorithms(True)
    else:
        torch.set_deterministic(True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    np.random.seed(seed=0)
    gpu_num = 0
    if use_cuda:
        gpu_num = torch.cuda.device_count()

    process_name = multiprocessing.current_process().name
    process_idx = int(process_name[process_name.rfind('-') + 1:])
    gpu_id = -1
    if gpu_num > 0:
        gpu_id = process_idx % gpu_num
    if gpu_id >= 0:
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"

    result = encode_one(args, device)
    result['model_idx'] = args['model_idx']
    return result


def filter_dict(result):
    keys = ['i_frame_num', 'p_frame_num',
            'ave_i_frame_bpp', 'ave_i_frame_psnr', 'ave_i_frame_msssim',
            'ave_p_frame_bpp', 'ave_p_frame_bpp_mv_y', 'ave_p_frame_bpp_mv_z', 'ave_p_frame_bpp_y',
            'ave_p_frame_bpp_z', 'ave_p_frame_psnr', 'ave_p_frame_msssim',
            'frame_bpp', 'frame_bpp_mv_y', 'frame_bpp_mv_z',
            'frame_bpp_y', 'frame_bpp_z', 'frame_psnr', 'frame_msssim', 'frame_type',
            'test_time']
    res = {k: v for k, v in result.items() if k in keys}

    i_num = res['i_frame_num']
    p_num = res['p_frame_num']
    sum_bits = i_num * res['ave_i_frame_bpp'] + p_num * res['ave_p_frame_bpp']
    sum_psnr = i_num * res['ave_i_frame_psnr'] + p_num * res['ave_p_frame_psnr']
    sum_msssim = i_num * res['ave_i_frame_msssim'] + p_num * res['ave_p_frame_msssim']
    res['ave_all_frame_bpp'] = sum_bits / (i_num + p_num)
    res['ave_all_frame_psnr'] = sum_psnr / (i_num + p_num)
    res['ave_all_frame_msssim'] = sum_msssim / (i_num + p_num)

    return res


def main():
    begin_time = time.time()

    torch.backends.cudnn.enabled = True
    args = parse_args()

    if args.cuda_device is not None and args.cuda_device != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    worker_num = args.worker
    assert worker_num >= 1

    with open(args.test_config) as f:
        config = json.load(f)

    multiprocessing.set_start_method("spawn")
    threadpool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_num)
    objs = []

    if args.force_intra:
        args.model_path = args.i_frame_model_path
    count_frames = 0
    count_sequences = 0
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        for seq_name in config[ds_name]['sequences']:
            count_sequences += 1
            for model_idx in range(len(args.model_path)):  # pylint: disable=C0200
                cur_dict = {}
                cur_dict['model_idx'] = model_idx
                cur_dict['i_frame_model_path'] = args.i_frame_model_path[model_idx]
                cur_dict['i_frame_model_name'] = args.i_frame_model_name
                cur_dict['force_intra'] = args.force_intra
                cur_dict['model_path'] = args.model_path[model_idx]
                cur_dict['video_path'] = seq_name
                cur_dict['gop'] = config[ds_name]['sequences'][seq_name]['gop']
                if args.force_intra:
                    cur_dict['gop'] = 1
                if args.force_intra_period > 0:
                    cur_dict['gop'] = args.force_intra_period
                cur_dict['frame_num'] = config[ds_name]['sequences'][seq_name]['frames']
                if args.force_frame_num > 0:
                    cur_dict['frame_num'] = args.force_frame_num
                cur_dict['dataset_path'] = config[ds_name]['base_path']
                cur_dict['write_stream'] = args.write_stream
                cur_dict['stream_path'] = f'{args.stream_path}/bit_stream'
                cur_dict['save_decoded_frame'] = args.save_decoded_frame
                cur_dict['decoded_frame_path'] = f'{args.decoded_frame_path}/decoded_video'
                cur_dict['ds_name'] = ds_name
                cur_dict['verbose'] = args.verbose

                count_frames += cur_dict['frame_num']

                obj = threadpool_executor.submit(
                    worker,
                    args.cuda,
                    cur_dict)
                objs.append(obj)

    results = []
    for obj in tqdm(objs):
        result = obj.result()
        results.append(result)

    log_result = {}

    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        log_result[ds_name] = {}
        for seq in config[ds_name]['sequences']:
            log_result[ds_name][seq] = {}
            for model in args.model_path:
                ckpt = os.path.basename(model)
                for res in results:
                    if res['name'].startswith(ckpt) and ds_name == res['ds_name'] \
                            and seq == res['video_path']:
                        log_result[ds_name][seq][ckpt] = filter_dict(res)

    out_json_dir = os.path.dirname(args.output_path)
    if len(out_json_dir) > 0:
        if not os.path.exists(out_json_dir):
            os.makedirs(out_json_dir)
    with open(args.output_path, 'w') as fp:
        json.dump(log_result, fp, indent=2)

    total_minutes = (time.time() - begin_time) / 60

    count_models = len(args.model_path)
    count_frames = count_frames // count_models
    print('Test finished')
    print(f'Tested {count_models} models on {count_frames} frames from {count_sequences} sequences')
    print(f'Total elapsed time: {total_minutes:.1f} min')


if __name__ == "__main__":
    main()
