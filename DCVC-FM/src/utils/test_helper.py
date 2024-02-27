# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

from src.models.video_model import DMC
from src.models.image_model import DMCI

from src.utils.common import str2bool, create_folder, generate_log_json
from src.utils.stream_helper import get_padding_size, get_state_dict, SPSHelper, NalType, \
    write_sps, read_header, read_sps_remaining, read_ip_remaining
from src.utils.video_reader import PNGReader, YUVReader
from src.utils.video_writer import PNGWriter, YUVWriter
from src.utils.metrics import calc_psnr, calc_msssim, calc_msssim_rgb
from src.transforms.functional import ycbcr444_to_420, ycbcr420_to_444, \
    rgb_to_ycbcr444, ycbcr444_to_rgb


def parse_args():
    parser = argparse.ArgumentParser(description="Example testing script")

    parser.add_argument("--ec_thread", type=str2bool, default=False)
    parser.add_argument("--stream_part_i", type=int, default=1)
    parser.add_argument("--stream_part_p", type=int, default=1)
    parser.add_argument('--model_path_i', type=str)
    parser.add_argument('--model_path_p',  type=str)
    parser.add_argument('--rate_num', type=int, default=4)
    parser.add_argument('--q_indexes_i', type=int, nargs="+")
    parser.add_argument('--q_indexes_p', type=int, nargs="+")
    parser.add_argument("--force_intra", type=str2bool, default=False)
    parser.add_argument("--force_frame_num", type=int, default=-1)
    parser.add_argument("--force_intra_period", type=int, default=-1)
    parser.add_argument("--rate_gop_size", type=int, default=8, choices=[4, 8])
    parser.add_argument('--reset_interval', type=int, default=32, required=False)
    parser.add_argument('--test_config', type=str, required=True)
    parser.add_argument('--force_root_path', type=str, default=None, required=False)
    parser.add_argument("--worker", "-w", type=int, default=1, help="worker number")
    parser.add_argument('--float16', type=str2bool, default=False)
    parser.add_argument("--cuda", type=str2bool, default=False)
    parser.add_argument('--cuda_idx', type=int, nargs="+", help='GPU indexes to use')
    parser.add_argument('--calc_ssim', type=str2bool, default=False, required=False)
    parser.add_argument('--write_stream', type=str2bool, default=False)
    parser.add_argument('--stream_path', type=str, default="out_bin")
    parser.add_argument('--save_decoded_frame', type=str2bool, default=False)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--verbose_json', type=str2bool, default=False)
    parser.add_argument('--verbose', type=int, default=0)

    args = parser.parse_args()
    return args


def np_image_to_tensor(img):
    image = torch.from_numpy(img).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    return image


def get_src_reader(args):
    if args['src_type'] == 'png':
        src_reader = PNGReader(args['src_path'], args['src_width'], args['src_height'])
    elif args['src_type'] == 'yuv420':
        src_reader = YUVReader(args['src_path'], args['src_width'], args['src_height'])
    return src_reader


def get_src_frame(args, src_reader, device):
    if args['src_type'] == 'yuv420':
        y, uv = src_reader.read_one_frame(dst_format="420")
        yuv = ycbcr420_to_444(y, uv)
        x = np_image_to_tensor(yuv)
        y = y[0, :, :]
        u = uv[0, :, :]
        v = uv[1, :, :]
        rgb = None
    else:
        assert args['src_type'] == 'png'
        rgb = src_reader.read_one_frame(dst_format="rgb")
        y, uv = rgb_to_ycbcr444(rgb)
        u, v = None, None
        yuv = np.concatenate((y, uv), axis=0)
        x = np_image_to_tensor(yuv)

    if args['float16']:
        x = x.to(torch.float16)
    x = x.to(device)
    return x, y, u, v, rgb


def get_distortion(args, x_hat, y, u, v, rgb):
    if args['src_type'] == 'yuv420':
        yuv_rec = x_hat.squeeze(0).cpu().numpy()
        y_rec, uv_rec = ycbcr444_to_420(yuv_rec)
        y_rec = y_rec[0, :, :]
        u_rec = uv_rec[0, :, :]
        v_rec = uv_rec[1, :, :]
        psnr_y = calc_psnr(y, y_rec, data_range=1)
        psnr_u = calc_psnr(u, u_rec, data_range=1)
        psnr_v = calc_psnr(v, v_rec, data_range=1)
        psnr = (6 * psnr_y + psnr_u + psnr_v) / 8
        if args['calc_ssim']:
            ssim_y = calc_msssim(y, y_rec, data_range=1)
            ssim_u = calc_msssim(u, u_rec, data_range=1)
            ssim_v = calc_msssim(v, v_rec, data_range=1)
        else:
            ssim_y, ssim_u, ssim_v = 0., 0., 0.
        ssim = (6 * ssim_y + ssim_u + ssim_v) / 8

        curr_psnr = [psnr, psnr_y, psnr_u, psnr_v]
        curr_ssim = [ssim, ssim_y, ssim_u, ssim_v]
    else:
        assert args['src_type'] == 'png'
        yuv_rec = x_hat.squeeze(0).cpu().numpy()
        rgb_rec = ycbcr444_to_rgb(yuv_rec[:1, :, :], yuv_rec[1:, :, :])
        psnr = calc_psnr(rgb, rgb_rec, data_range=1)
        if args['calc_ssim']:
            msssim = calc_msssim_rgb(rgb, rgb_rec, data_range=1)
        else:
            msssim = 0.
        curr_psnr = [psnr]
        curr_ssim = [msssim]
    return curr_psnr, curr_ssim


def run_one_point_fast(p_frame_net, i_frame_net, args):
    frame_num = args['frame_num']
    rate_gop_size = args['rate_gop_size']
    verbose = args['verbose']
    reset_interval = args['reset_interval']
    verbose_json = args['verbose_json']
    device = next(i_frame_net.parameters()).device

    frame_types = []
    psnrs = []
    msssims = []
    bits = []
    index_map = [0, 1, 0, 2, 0, 2, 0, 2]

    start_time = time.time()
    src_reader = get_src_reader(args)
    pic_height = args['src_height']
    pic_width = args['src_width']
    padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height, pic_width, 16)

    with torch.no_grad():
        for frame_idx in range(frame_num):
            frame_start_time = time.time()
            x, y, u, v, rgb = get_src_frame(args, src_reader, device)

            # pad if necessary
            x_padded = F.pad(x, (padding_l, padding_r, padding_t, padding_b), mode="replicate")

            if frame_idx % args['intra_period'] == 0:
                result = i_frame_net.encode(x_padded, args['q_index_i'])
                dpb = {
                    "ref_frame": result["x_hat"],
                    "ref_feature": None,
                    "ref_mv_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None,
                }
                recon_frame = result["x_hat"]
                frame_types.append(0)
                bits.append(result["bit"])
            else:
                if reset_interval > 0 and frame_idx % reset_interval == 1:
                    dpb["ref_feature"] = None
                fa_idx = index_map[frame_idx % rate_gop_size]
                result = p_frame_net.encode(x_padded, dpb, args['q_index_p'], fa_idx)

                dpb = result["dpb"]
                recon_frame = dpb["ref_frame"]
                frame_types.append(1)
                bits.append(result['bit'])

            recon_frame = recon_frame.clamp_(0, 1)
            x_hat = F.pad(recon_frame, (-padding_l, -padding_r, -padding_t, -padding_b))
            frame_end_time = time.time()
            curr_psnr, curr_ssim = get_distortion(args, x_hat, y, u, v, rgb)
            psnrs.append(curr_psnr)
            msssims.append(curr_ssim)

            if verbose >= 2:
                print(f"frame {frame_idx}, {frame_end_time - frame_start_time:.3f} seconds, "
                      f"bits: {bits[-1]:.3f}, PSNR: {psnrs[-1][0]:.4f}, "
                      f"MS-SSIM: {msssims[-1][0]:.4f} ")

    src_reader.close()
    test_time = time.time() - start_time

    log_result = generate_log_json(frame_num, pic_height * pic_width, test_time,
                                   frame_types, bits, psnrs, msssims, verbose=verbose_json)
    return log_result


def run_one_point_with_stream(p_frame_net, i_frame_net, args):
    frame_num = args['frame_num']
    rate_gop_size = args['rate_gop_size']
    save_decoded_frame = args['save_decoded_frame']
    verbose = args['verbose']
    reset_interval = args['reset_interval']
    verbose_json = args['verbose_json']
    device = next(i_frame_net.parameters()).device

    src_reader = get_src_reader(args)
    pic_height = args['src_height']
    pic_width = args['src_width']
    padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height, pic_width, 16)

    frame_types = []
    psnrs = []
    msssims = []
    bits = []

    start_time = time.time()
    p_frame_number = 0
    overall_p_encoding_time = 0
    overall_p_decoding_time = 0
    index_map = [0, 1, 0, 2, 0, 2, 0, 2]

    bitstream_path = Path(args['curr_bin_path'])
    output_file = bitstream_path.open("wb")
    sps_helper = SPSHelper()
    outstanding_sps_bytes = 0
    sps_buffer = []

    with torch.no_grad():
        for frame_idx in range(frame_num):
            frame_start_time = time.time()
            x, y, u, v, rgb = get_src_frame(args, src_reader, device)

            # pad if necessary
            x_padded = F.pad(x, (padding_l, padding_r, padding_t, padding_b), mode="replicate")

            if frame_idx % args['intra_period'] == 0:
                sps = {
                    'sps_id': -1,
                    'height': pic_height,
                    'width': pic_width,
                    'qp': args['q_index_i'],
                    'fa_idx': 0,
                }
                sps_id, sps_new = sps_helper.get_sps_id(sps)
                sps['sps_id'] = sps_id
                if sps_new:
                    outstanding_sps_bytes += write_sps(output_file, sps)
                    if verbose >= 2:
                        print("new sps", sps)
                result = i_frame_net.encode(x_padded, args['q_index_i'], sps_id, output_file)
                dpb = {
                    "ref_frame": result["x_hat"],
                    "ref_feature": None,
                    "ref_mv_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None,
                }
                recon_frame = result["x_hat"]
                frame_types.append(0)
                bits.append(result["bit"] + outstanding_sps_bytes * 8)
                outstanding_sps_bytes = 0
            else:
                fa_idx = index_map[frame_idx % rate_gop_size]
                if reset_interval > 0 and frame_idx % reset_interval == 1:
                    dpb["ref_feature"] = None
                    fa_idx = 3

                sps = {
                    'sps_id': -1,
                    'height': pic_height,
                    'width': pic_width,
                    'qp': args['q_index_p'],
                    'fa_idx': fa_idx,
                }
                sps_id, sps_new = sps_helper.get_sps_id(sps)
                sps['sps_id'] = sps_id
                if sps_new:
                    outstanding_sps_bytes += write_sps(output_file, sps)
                    if verbose >= 2:
                        print("new sps", sps)
                result = p_frame_net.encode(x_padded, dpb, args['q_index_p'], fa_idx, sps_id,
                                            output_file)

                dpb = result["dpb"]
                recon_frame = dpb["ref_frame"]
                frame_types.append(1)
                bits.append(result['bit'] + outstanding_sps_bytes * 8)
                outstanding_sps_bytes = 0
                p_frame_number += 1
                overall_p_encoding_time += result['encoding_time']

            recon_frame = recon_frame.clamp_(0, 1)
            x_hat = F.pad(recon_frame, (-padding_l, -padding_r, -padding_t, -padding_b))
            frame_end_time = time.time()
            curr_psnr, curr_ssim = get_distortion(args, x_hat, y, u, v, rgb)
            psnrs.append(curr_psnr)
            msssims.append(curr_ssim)

            if verbose >= 2:
                print(f"frame {frame_idx} encoded, {frame_end_time - frame_start_time:.3f} s, "
                      f"bits: {bits[-1]}, PSNR: {psnrs[-1][0]:.4f}, "
                      f"MS-SSIM: {msssims[-1][0]:.4f} ")

    src_reader.close()
    output_file.close()
    sps_helper = SPSHelper()
    input_file = bitstream_path.open("rb")
    decoded_frame_number = 0
    src_reader = get_src_reader(args)

    if save_decoded_frame:
        if args['src_type'] == 'png':
            recon_writer = PNGWriter(args['bin_folder'], args['src_width'], args['src_height'])
        elif args['src_type'] == 'yuv420':
            recon_writer = YUVWriter(args['curr_rec_path'], args['src_width'], args['src_height'])
    pending_frame_spss = []
    with torch.no_grad():
        while decoded_frame_number < frame_num:
            new_stream = False
            if len(pending_frame_spss) == 0:
                header = read_header(input_file)
                if header['nal_type'] == NalType.NAL_SPS:
                    sps = read_sps_remaining(input_file, header['sps_id'])
                    sps_helper.add_sps_by_id(sps)
                    if verbose >= 2:
                        print("new sps", sps)
                    continue
                if header['nal_type'] == NalType.NAL_Ps:
                    pending_frame_spss = header['sps_ids'][1:]
                    sps_id = header['sps_ids'][0]
                else:
                    sps_id = header['sps_id']
                new_stream = True
            else:
                sps_id = pending_frame_spss[0]
                pending_frame_spss.pop(0)
            sps = sps_helper.get_sps_by_id(sps_id)
            if new_stream:
                bit_stream = read_ip_remaining(input_file)
            else:
                bit_stream = None
            frame_start_time = time.time()
            x, y, u, v, rgb = get_src_frame(args, src_reader, device)
            if header['nal_type'] == NalType.NAL_I:
                decoded = i_frame_net.decompress(bit_stream, sps)
                dpb = {
                    "ref_frame": decoded["x_hat"],
                    "ref_feature": None,
                    "ref_mv_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None,
                }
                recon_frame = decoded["x_hat"]
            elif header['nal_type'] == NalType.NAL_P or header['nal_type'] == NalType.NAL_Ps:
                if sps['fa_idx'] == 3:
                    dpb["ref_feature"] = None
                decoded = p_frame_net.decompress(bit_stream, dpb, sps)
                dpb = decoded["dpb"]
                recon_frame = dpb["ref_frame"]
                overall_p_decoding_time += decoded['decoding_time']

            recon_frame = recon_frame.clamp_(0, 1)
            x_hat = F.pad(recon_frame, (-padding_l, -padding_r, -padding_t, -padding_b))
            frame_end_time = time.time()
            curr_psnr, curr_ssim = get_distortion(args, x_hat, y, u, v, rgb)
            assert psnrs[decoded_frame_number][0] == curr_psnr[0]

            if verbose >= 2:
                stream_length = 0 if bit_stream is None else len(bit_stream) * 8
                print(f"frame {decoded_frame_number} decoded, "
                      f"{frame_end_time - frame_start_time:.3f} s, "
                      f"bits: {stream_length}, PSNR: {curr_psnr[0]:.4f} ")

            if save_decoded_frame:
                yuv_rec = x_hat.squeeze(0).cpu().numpy()
                if args['src_type'] == 'yuv420':
                    y_rec, uv_rec = ycbcr444_to_420(yuv_rec)
                    recon_writer.write_one_frame(y=y_rec, uv=uv_rec, src_format='420')
                else:
                    assert args['src_type'] == 'png'
                    rgb_rec = ycbcr444_to_rgb(yuv_rec[:1, :, :], yuv_rec[1:, :, :])
                    recon_writer.write_one_frame(rgb=rgb_rec, src_format='rgb')
            decoded_frame_number += 1
    input_file.close()
    src_reader.close()

    if save_decoded_frame:
        recon_writer.close()

    test_time = time.time() - start_time
    if verbose >= 1 and p_frame_number > 0:
        print(f"encoding/decoding {p_frame_number} P frames, "
              f"average encoding time {overall_p_encoding_time/p_frame_number * 1000:.0f} ms, "
              f"average decoding time {overall_p_decoding_time/p_frame_number * 1000:.0f} ms.")

    log_result = generate_log_json(frame_num, pic_height * pic_width, test_time,
                                   frame_types, bits, psnrs, msssims, verbose=verbose_json)
    with open(args['curr_json_path'], 'w') as fp:
        json.dump(log_result, fp, indent=2)
    return log_result


i_frame_net = None  # the model is initialized after each process is spawn, thus OK for multiprocess
p_frame_net = None


def worker(args):
    global i_frame_net
    global p_frame_net

    sub_dir_name = args['seq']
    bin_folder = os.path.join(args['stream_path'], args['ds_name'])
    if args['write_stream']:
        create_folder(bin_folder, True)

    args['src_path'] = os.path.join(args['dataset_path'], sub_dir_name)
    args['bin_folder'] = bin_folder
    args['curr_bin_path'] = os.path.join(bin_folder,
                                         f"{args['seq']}_q{args['q_index_i']}.bin")
    args['curr_rec_path'] = args['curr_bin_path'].replace('.bin', '.yuv')
    args['curr_json_path'] = args['curr_bin_path'].replace('.bin', '.json')

    if args['write_stream']:
        result = run_one_point_with_stream(p_frame_net, i_frame_net, args)
    else:
        result = run_one_point_fast(p_frame_net, i_frame_net, args)

    result['ds_name'] = args['ds_name']
    result['seq'] = args['seq']
    result['rate_idx'] = args['rate_idx']

    return result


def init_func(args, gpu_num):
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    np.random.seed(seed=0)

    process_name = multiprocessing.current_process().name
    process_idx = int(process_name[process_name.rfind('-') + 1:])
    gpu_id = -1
    if gpu_num > 0:
        gpu_id = process_idx % gpu_num
    if gpu_id >= 0:
        if args.cuda_idx is not None:
            gpu_id = args.cuda_idx[gpu_id]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = "cuda:0"
    else:
        device = "cpu"

    global i_frame_net
    i_state_dict = get_state_dict(args.model_path_i)
    i_frame_net = DMCI(ec_thread=args.ec_thread, stream_part=args.stream_part_i, inplace=True)
    i_frame_net.load_state_dict(i_state_dict)
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()

    global p_frame_net
    if not args.force_intra:
        p_state_dict = get_state_dict(args.model_path_p)
        p_frame_net = DMC(ec_thread=args.ec_thread, stream_part=args.stream_part_p, inplace=True)
        p_frame_net.load_state_dict(p_state_dict)
        p_frame_net = p_frame_net.to(device)
        p_frame_net.eval()

    if args.write_stream:
        if p_frame_net is not None:
            p_frame_net.update(force=True)
        i_frame_net.update(force=True)

    if args.float16:
        if p_frame_net is not None:
            p_frame_net.half()
        i_frame_net.half()
