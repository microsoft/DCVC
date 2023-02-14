# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import concurrent.futures
import json
import multiprocessing
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from src.models import build_model
from src.utils.common import str2bool, interpolate_log, create_folder, dump_json
from src.utils.stream_helper import get_padding_size, get_state_dict
from src.utils.png_reader import PNGReader
from tqdm import tqdm
from pytorch_msssim import ms_ssim


def parse_args():
    parser = argparse.ArgumentParser(description="Example testing script")
    parser.add_argument("--ec_thread", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--i_frame_model', type=str, default='EncSDecS')
    parser.add_argument('--i_frame_model_path', type=str, default='')
    parser.add_argument('--i_frame_q_scales', type=float, nargs="+")
    parser.add_argument('--rate_num', type=int, default=4)
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


def np_image_to_tensor(img):
    image = torch.from_numpy(img).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    return image


def save_torch_image(img, save_path):
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)


def PSNR(input1, input2):
    mse = torch.mean((input1 - input2) ** 2)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr.item()


def avg_imagejson(video_json):
    dirname = os.path.dirname(video_json)
    basename = os.path.basename(video_json)
    names = basename.split('.')
    names[-2] = names[-2] + '_avg'
    out_json_file = os.path.join(dirname, '.'.join(names))
    out_json = dict()
    out_json['name'] = str(video_json)

    def RD_oneclass(videojson):
        bpp_d = dict()
        psnr_d = dict()
        rate_d = dict()
        for subseq in videojson:
            sub = videojson[subseq]
            for image in sub:
                img = sub[image]
                if 'rate_idx' in img:
                    rate_idx = img['rate_idx']
                elif 'qp' in img:
                    rate_idx = img['qp']
                bpp = img['bpp']
                psnr = img['psnr']
                bpp_d[rate_idx] = bpp_d.get(rate_idx, 0) + bpp
                psnr_d[rate_idx] = psnr_d.get(rate_idx, 0) + psnr
                rate_d[rate_idx] = rate_d.get(rate_idx, 0) + 1
        rates = list(rate_d.keys())
        rates.sort()
        bpp = []
        psnr = []
        for rate in rates:
            bpp.append(bpp_d[rate] / rate_d[rate])
            psnr.append(psnr_d[rate] / rate_d[rate])
        return bpp, psnr

    with open(video_json) as f:
        videojson = json.load(f)

    for c in videojson:
        bpp, psnr = RD_oneclass(videojson[c])
        out_json[c] = dict()
        out_json[c]['bpp'] = bpp
        out_json[c]['psnr'] = psnr

    with open(out_json_file, 'w') as fp:
        dump_json(out_json, fp, float_digits=6, indent=2)


def run_test(i_frame_net, args, device):
    write_stream = 'write_stream' in args and args['write_stream']
    save_decoded_frame = 'save_decoded_frame' in args and args['save_decoded_frame']
    verbose = args['verbose'] if 'verbose' in args else 0

    if args['src_type'] == 'png' and args['img_folder'] == 1:
        src_reader = PNGReader(args['img_path'])
    else:
        raise NotImplementedError()

    frame_pixel_num = 0

    start_time = time.time()
    with torch.no_grad():
        frame_start_time = time.time()
        rgb = src_reader.read_one_frame(src_format="rgb")
        x = np_image_to_tensor(rgb)
        x = x.to(device)
        pic_height = x.shape[2]
        pic_width = x.shape[3]

        if frame_pixel_num == 0:
            frame_pixel_num = x.shape[2] * x.shape[3]
        else:
            assert frame_pixel_num == x.shape[2] * x.shape[3]

        # pad if necessary
        padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height, pic_width)
        x_padded = torch.nn.functional.pad(
            x,
            (padding_l, padding_r, padding_t, padding_b),
            mode="constant",
            value=0,
        )

        bin_path = os.path.join(args['bin_folder'], "0.bin") \
            if write_stream else None

        result = i_frame_net.encode_decode(x_padded, args['i_frame_q_scale'], bin_path,
                                           pic_height=pic_height, pic_width=pic_width)

        bits = result["bit"]
        bpp = bits / pic_width / pic_height
        encoding_time = result.get('enc_time', 0)
        decoding_time = result.get('dec_time', 0)
        recon_frame = result["x_hat"]
        recon_frame = recon_frame.clamp_(0, 1)
        x_hat = F.pad(recon_frame, (-padding_l, -padding_r, -padding_t, -padding_b))
        psnr = PSNR(x_hat, x)
        msssim = ms_ssim(x_hat, x, data_range=1).item()
        frame_end_time = time.time()

        if verbose >= 1:
            print(f"{os.path.basename(args['img_path'])}, rate: {args['rate_idx']}, "
                  f"{frame_end_time - frame_start_time:.3f} seconds,",
                  f"bpp: {bpp:.3f}, PSNR: {psnr:.4f}, MS-SSIM: {msssim:.4f}, "
                  f"encoding_time: {encoding_time:.4f}, decoding_time: {decoding_time:.4f} ")

        if save_decoded_frame:
            folder_name = f"{args['rate_idx']}_{bpp:.4f}_{psnr:.4f}_{encoding_time:.4f}_{decoding_time:.4f}"
            save_path = os.path.join(args['decoded_frame_folder'], f"{os.path.basename(args['img_path'])}.png")
            save_torch_image(x_hat, save_path)
            os.rename(args['decoded_frame_folder'], args['decoded_frame_folder'] + f'/../{folder_name}')

    test_time = time.time() - start_time

    log_result = {}
    log_result['frame_pixel_num'] = frame_pixel_num
    log_result['bpp'] = bpp
    log_result['psnr'] = psnr
    log_result['msssim'] = msssim
    log_result['test_time'] = test_time
    return log_result


def encode_one(args, device):
    i_state_dict = get_state_dict(args['i_frame_model_path'])
    i_frame_net = build_model(args['i_frame_model'], ec_thread=args['ec_thread'])
    i_frame_net.load_state_dict(i_state_dict, verbose=False)
    if hasattr(i_frame_net, 'set_rate'):
        i_frame_net.set_rate(args['rate_idx'])
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()

    if args['write_stream']:
        i_frame_net.update(force=True)

    sub_dir_name = os.path.basename(args['img_path'])

    bin_folder = os.path.join(args['stream_path'], sub_dir_name, str(args['rate_idx']))
    if args['write_stream']:
        create_folder(bin_folder, True)

    if args['save_decoded_frame']:
        decoded_frame_folder = os.path.join(args['decoded_frame_path'], sub_dir_name,
                                            str(args['rate_idx']))
        create_folder(decoded_frame_folder)
    else:
        decoded_frame_folder = None

    if 'img_path' not in args:
        args['img_path'] = os.path.join(args['dataset_path'], sub_dir_name)
    args['bin_folder'] = bin_folder
    args['decoded_frame_folder'] = decoded_frame_folder

    result = run_test(i_frame_net, args, device=device)

    result['ds_name'] = args['ds_name']
    result['img_path'] = args['img_path']
    result['rate_idx'] = args['rate_idx']

    return result


def worker(use_cuda, args):
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        torch.set_deterministic(True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    np.random.seed(seed=0)
    gpu_num = 0
    if use_cuda:
        gpu_num = torch.cuda.device_count()

    process_name = multiprocessing.current_process().name
    if process_name == 'MainProcess':
        process_idx = 0
    else:
        process_idx = int(process_name[process_name.rfind('-') + 1:])
    gpu_id = -1
    if gpu_num > 0:
        gpu_id = process_idx % gpu_num
    if gpu_id >= 0:
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"

    result = encode_one(args, device)
    return result


def prepare_args(args, config, ds_name, img_name, rate_idx, i_frame_q_scales):
    cur_args = {}
    cur_args['rate_idx'] = rate_idx
    cur_args['ec_thread'] = args.ec_thread
    cur_args['i_frame_model'] = args.i_frame_model
    cur_args['i_frame_model_path'] = args.i_frame_model_path
    if len(i_frame_q_scales) > 0:
        cur_args['i_frame_q_scale'] = i_frame_q_scales[rate_idx].to(torch.float32)
    else:
        cur_args['i_frame_q_scale'] = []
    cur_args['img_path'] = img_name
    cur_args['src_type'] = config[ds_name]['src_type']
    cur_args['img_folder'] = config[ds_name].get('img_folder', 0)
    if cur_args['img_folder'] == 1:
        cur_args['img_path'] = os.path.join(args.root_path, config[ds_name]['base_path'], img_name)
    else:
        cur_args['src_height'] = config[ds_name]['sequences'][img_name]['height']
        cur_args['src_width'] = config[ds_name]['sequences'][img_name]['width']
        cur_args['dataset_path'] = os.path.join(args.root_path, config[ds_name]['base_path'])
    cur_args['write_stream'] = args.write_stream
    cur_args['stream_path'] = args.stream_path
    cur_args['save_decoded_frame'] = args.save_decoded_frame
    cur_args['decoded_frame_path'] = f'{args.decoded_frame_path}'
    cur_args['ds_name'] = ds_name
    cur_args['verbose'] = args.verbose
    return cur_args


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

    if worker_num > 1:
        multiprocessing.set_start_method("spawn")
        threadpool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_num)
        objs = []

    count_frames = 0
    count_imgs = 0

    rate_num = args.rate_num
    q_scales_list = []

    ckpt = get_state_dict(args.i_frame_model_path)
    if "q_scale" in ckpt:
        q_scales = ckpt["q_scale"]
    elif "student.q_scale" in ckpt:
        q_scales = ckpt["student.q_scale"]
    elif "teacher.q_scale" in ckpt:
        q_scales = ckpt["teacher.q_scale"]
    else:
        raise ValueError("q_scale")
    q_scales_list.append(q_scales.reshape(-1))
    if q_scales_list:
        i_frame_q_scales = torch.cat(q_scales_list)
    else:
        i_frame_q_scales = []

    print("q_scales in intra ckpt: ", end='')
    for q in i_frame_q_scales:
        print(f"{q:.3f}, ", end='')
    print()
    if args.i_frame_q_scales is not None:
        assert len(args.i_frame_q_scales) == rate_num
        i_frame_q_scales = torch.tensor(args.i_frame_q_scales)
        print(f"testing {rate_num} rate points with pre-defined intra y q_scales: ", end='')
    elif len(i_frame_q_scales) == rate_num:
        print(f"testing {rate_num} rate points with intra y q_scales in ckpt: ", end='')
    elif len(i_frame_q_scales) > 0:
        max_q_scale = i_frame_q_scales[0]
        min_q_scale = i_frame_q_scales[-1]
        i_frame_q_scales = interpolate_log(min_q_scale, max_q_scale, rate_num)
        i_frame_q_scales = torch.tensor(i_frame_q_scales)
        print(f"testing {rate_num} rates, using intra y q_scales: ", end='')

    for q in i_frame_q_scales:
        print(f"{q:.3f}, ", end='')
    print()

    results = []
    args.root_path = config['root_path']
    config = config['test_classes']
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        if config[ds_name].get('img_folder', 0) == 1:
            imgs = os.listdir(os.path.join(args.root_path, config[ds_name]['base_path']))
            imgs.sort()
            imgs = [f for f in imgs if f.endswith(config[ds_name]['src_type'])]
        else:
            imgs = config[ds_name]['sequences']
        for img_name in imgs:
            count_imgs += 1
            for rate_idx in range(rate_num):
                cur_args = prepare_args(args, config, ds_name, img_name, rate_idx, i_frame_q_scales)
                count_frames += 1

                if worker_num > 1:
                    obj = threadpool_executor.submit(
                        worker,
                        args.cuda,
                        cur_args)
                    objs.append(obj)
                else:
                    result = worker(args.cuda, cur_args)
                    results.append(result)

    if worker_num > 1:
        for obj in tqdm(objs):
            result = obj.result()
            results.append(result)

    log_result = {}
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        log_result[ds_name] = {}
        if config[ds_name].get('img_folder', 0) == 1:
            imgs = os.listdir(os.path.join(args.root_path, config[ds_name]['base_path']))
            imgs.sort()
            imgs = [f for f in imgs if f.endswith(config[ds_name]['src_type'])]
        else:
            imgs = config[ds_name]['sequences']
        for img in imgs:
            log_result[ds_name][img] = {}
            for rate in range(rate_num):
                for res in results:
                    if res['rate_idx'] == rate and ds_name == res['ds_name'] \
                            and img == os.path.basename(res['img_path']):
                        log_result[ds_name][img][f"{rate:03d}"] = res

    out_json_dir = os.path.dirname(args.output_path)
    if len(out_json_dir) > 0:
        create_folder(out_json_dir, True)
    with open(args.output_path, 'w') as fp:
        dump_json(log_result, fp, float_digits=6, indent=2)

    avg_imagejson(args.output_path)

    total_minutes = (time.time() - begin_time) / 60
    print('Test finished')
    print(f'Tested {count_frames} frames from {count_imgs} images')
    print(f'Total elapsed time: {total_minutes:.1f} min')


if __name__ == "__main__":
    main()
