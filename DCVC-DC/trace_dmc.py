from src.tetra.model_wrapper import *
import torch
import torch.nn as nn

model = DMC_wrapper(model_path="./checkpoints/cvpr2023_video_yuv420_psnr.pth.tar")
model.eval()

shape_map = {
    "1080p" : (1, 3, 1088, 1920),
    "720p" : (1, 3, 720, 1280),
    "360p" : (1, 3, 368, 480)
}

for model_type in ["360p"]:

    x = torch.rand(shape_map[model_type])
    ref_frame = x

    traced_model = torch.jit.trace(model, (x, ref_frame), check_trace=False)
    torch.jit.save(traced_model, f"dmc_traced_{model_type}.jit")
