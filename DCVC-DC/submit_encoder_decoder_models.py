from src.tetra.model_wrapper import *
import os
import torch
import tetra_hub as hub


def submit_job(model, name, input_shapes):
    devices = [
        hub.Device(name="Apple iPhone 14 Pro"),
        hub.Device(name="Apple iPhone 12 Pro")
    ]
    for each in devices:
        hub.submit_profile_job(model=model, name=name, input_shapes=input_shapes, device=each, options="--enable_mlpackage")

cwd = "./"
image_model_path = "./checkpoints/cvpr2023_image_yuv420_psnr.pth.tar"
video_model_path = "./checkpoints/cvpr2023_video_yuv420_psnr.pth.tar"
intra_no_ar_encoder = IntraNoAR_encoder_wrapper(model_path=os.path.join(cwd, image_model_path))
intra_no_ar_decoder = IntraNoAR_decoder_wrapper(model_path=os.path.join(cwd, image_model_path))

dmc_encoder = DMC_encoder_wrapper(model_path=cwd+video_model_path)
dmc_decoder = DMC_decoder_wrapper(model_path=cwd+video_model_path)

shape_map = {
    "1080p" : (1, 3, 1088, 1920),
    "720p" : (1, 3, 720, 1280),
    "360p" : (1, 3, 368, 480)
}

# dummy input to ensure q_index is not constant folded
sample_q_index = torch.Tensor([0]).reshape(1,).type(torch.long)
dummy_input = torch.Tensor([0]).reshape(1,)

# input shapes
encoder_input = (1, 3, 368, 480)
intra_no_ar_y_hat = (1, 256, 23, 30)
dmc_y_hat = (1, 128, 23, 30)
dmc_c1 = (1, 48, 368, 480)
dmc_c2 = (1, 64, 184, 240)
dmc_c3 = (1, 96, 92, 120)

# trace and convert IntraNoAR models

x = torch.rand(encoder_input)
traced_model = torch.jit.trace(intra_no_ar_encoder, (x, sample_q_index, dummy_input), check_trace=False)
submit_job(traced_model, "IntraNoAR_Encoder", {"x" : x.shape, "q_index": sample_q_index.shape, "dummy_input": dummy_input.shape})

y_hat = torch.rand(intra_no_ar_y_hat)
traced_model = torch.jit.trace(intra_no_ar_decoder, (y_hat, sample_q_index, dummy_input), check_trace=False)
submit_job(traced_model, "IntraNoAR_Decoder", {"x" : y_hat.shape, "q_index": sample_q_index.shape, "dummy_input": dummy_input.shape})

#
# trace and convert DMC models
#

sample_frame_idx = torch.Tensor([0]).reshape(1,).type(torch.long)
x = torch.rand(encoder_input)
ref_frame = x

traced_model = torch.jit.trace(dmc_encoder, (x, ref_frame, sample_q_index, sample_frame_idx, dummy_input), check_trace=False)
submit_job(traced_model, "DMC_Encoder", {"x" : x.shape, "ref_frame": x.shape, "q_index": sample_q_index.shape, "frame_idx": sample_frame_idx.shape, "dummy_input": dummy_input.shape})


y_hat = torch.randn(dmc_y_hat)
dmc_c1 = torch.randn(dmc_c1)
dmc_c2 = torch.randn(dmc_c2)
dmc_c3 = torch.randn(dmc_c3)

traced_model = torch.jit.trace(dmc_decoder, (y_hat, dmc_c1, dmc_c2, dmc_c3, sample_q_index, dummy_input), check_trace=False)
submit_job(traced_model, "DMC_Decoder", {"y_hat" : y_hat.shape, "c1": dmc_c1.shape, "c2": dmc_c2.shape, "c3": dmc_c3.shape, "q_index": sample_q_index.shape, "dummy_input": dummy_input.shape})

