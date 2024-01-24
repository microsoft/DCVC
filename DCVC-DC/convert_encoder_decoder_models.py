from src.tetra.model_wrapper import *

import argparse
import os
import torch

from coremltools_extensions import convert as cte_convert
import coremltools as ct

def convert_with_extesions(traced_model, model_name, output_dir, input_shapes, convert_to="mlpackage", os_version="16.2"):
    """
        Converts traced torch model into coreml model format
        traced_model: Traced torch model
        model_name: name of the converted model to set during serialization (without extensions)
        output_dir: output dir to save model to
        input_shapes: List of tuple of input shapes in order
        convert_to: Convert to "mlpackage" or "neuralnetwork" format of CoreML. Default: "mlpackage"
        os_version: target OS version. Setting this to 13, 14, 15, 16 triggers related graph passes
            during conversion. Default: "16.2"
    """
    ct_inputs = [ ct.TensorType(shape=shape) for shape in input_shapes ]
    model = cte_convert(traced_model, inputs=ct_inputs, os_version=os_version)
    output_path = os.path.join(output_dir, model_name)
    model.save(f"{output_path}.mlpackage")


# Example usage:
# python convert_encoder_decoder_models.py --i_frame_model_path ./checkpoints/cvpr2023_image_yuv420_psnr.pth.tar --p_frame_model_path ./checkpoints/cvpr2023_video_yuv420_psnr.pth.tar --output_assets_dir ./converted_coreml_assets
#
parser = argparse.ArgumentParser(description="Converting DCVC models to CoreML model format")
parser.add_argument(
    "--output_assets_dir", type=str, required=True
)
parser.add_argument(
    "--i_frame_model_path", type=str, required=True
)
parser.add_argument(
    "--p_frame_model_path", type=str, required=True
)
args = parser.parse_args()

asset_dir = args.output_assets_dir
image_model_path = args.i_frame_model_path
video_model_path = args.p_frame_model_path

os.makedirs(asset_dir, exist_ok=True)

# Load IntraNoAR model wrappers
intra_no_ar_encoder = IntraNoAR_encoder_wrapper(model_path=image_model_path)
intra_no_ar_decoder = IntraNoAR_decoder_wrapper(model_path=image_model_path)

# Load DMC model wrappers
dmc_encoder = DMC_encoder_wrapper(model_path=video_model_path)
dmc_decoder = DMC_decoder_wrapper(model_path=video_model_path)

# dummy input to ensure q_index is not constant folded
sample_q_index = torch.Tensor([0]).reshape(1,).type(torch.long)
dummy_input = torch.Tensor([0]).reshape(1,)

# Following input shapes are computed w.r.t 360p model
# and considering above model wrappers
# NOTE: If model wrappers are updated, these input shapes
# should reflect new inputs and their respective shapes

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
#submit_job(traced_model, "IntraNoAR_Encoder", {"x" : x.shape, "q_index": sample_q_index.shape, "dummy_input": dummy_input.shape})
convert_with_extesions(traced_model=traced_model, model_name="IntraNoAR_encoder", output_dir=asset_dir, input_shapes=[x.shape, sample_q_index.shape, dummy_input.shape])

y_hat = torch.rand(intra_no_ar_y_hat)
traced_model = torch.jit.trace(intra_no_ar_decoder, (y_hat, sample_q_index, dummy_input), check_trace=False)
#submit_job(traced_model, "IntraNoAR_Decoder", {"x" : y_hat.shape, "q_index": sample_q_index.shape, "dummy_input": dummy_input.shape})
convert_with_extesions(traced_model=traced_model, model_name="IntraNoAR_decoder", output_dir=asset_dir, input_shapes=[y_hat.shape, sample_q_index.shape, dummy_input.shape])

#
# trace and convert DMC models
#

sample_frame_idx = torch.Tensor([0]).reshape(1,).type(torch.long)
x = torch.rand(encoder_input)
ref_frame = x

traced_model = torch.jit.trace(dmc_encoder, (x, ref_frame, sample_q_index, sample_frame_idx, dummy_input), check_trace=False)
convert_with_extesions(traced_model=traced_model, model_name="DMC_encoder", output_dir=asset_dir, input_shapes=[x.shape, x.shape, sample_q_index.shape, sample_frame_idx.shape, dummy_input.shape])

y_hat = torch.randn(dmc_y_hat)
dmc_c1 = torch.randn(dmc_c1)
dmc_c2 = torch.randn(dmc_c2)
dmc_c3 = torch.randn(dmc_c3)

traced_model = torch.jit.trace(dmc_decoder, (y_hat, dmc_c1, dmc_c2, dmc_c3, sample_q_index, dummy_input), check_trace=False)
convert_with_extesions(traced_model=traced_model, model_name="DMC_decoder", output_dir=asset_dir, input_shapes=[y_hat.shape, dmc_c1.shape, dmc_c2.shape, dmc_c3.shape, sample_q_index.shape, dummy_input.shape])


