from ..models.image_model import IntraNoAR
from ..models.video_model import DMC
from ..utils.stream_helper import get_state_dict

import torch.nn as nn

## IntraNoAr wrapper
class IntraNoAR_wrapper(nn.Module):
    def __init__(self, model_path, mode="forward", N=256, anchor_num=4, ec_thread=False, stream_part=1, inplace=False, q_in_ckpt=False):
        super().__init__()
        self.model = IntraNoAR(N, anchor_num, ec_thread, stream_part, inplace)
        state_dict = get_state_dict(model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        assert mode in { "forward", "encoder", "decoder"}
        self.mode = mode
        self.torch_output_order = None
        self.q_in_ckpt = q_in_ckpt

    def forward(self, x, q_index, dummy_input):
        out_dict = self.model(x, q_in_ckpt=self.q_in_ckpt, q_index=q_index, dummy_input=dummy_input)
        return out_dict["x_hat"], out_dict["bit"], out_dict["bpp"], out_dict["bpp_y"], out_dict["bpp_z"]

# TODO: IntraNoAR encoder
# Please update forward pass here to mimic how encoder from image_model.py
# with-out framework code e.g. encoder stream.
class IntraNoAR_encoder_wrapper(IntraNoAR_wrapper):
    def __init__(self, model_path, mode="encoder", N=256, anchor_num=4, ec_thread=False, stream_part=1, inplace=False, q_in_ckpt=False):
        super().__init__(model_path=model_path, mode=mode, N=N, anchor_num=anchor_num, ec_thread=ec_thread, stream_part=stream_part, inplace=inplace, q_in_ckpt=q_in_ckpt)

    def forward(self, x, q_index, dummy_input):
        out_dict = self.model.compress_without_entropy_coder(x, self.q_in_ckpt, q_index, dummy_input)
        # Unfold dictionary and return elements individually
        return out_dict["y_hat"], out_dict["z_hat"], out_dict["y_q_w_0"], out_dict["y_q_w_1"], out_dict["y_q_w_2"], out_dict["y_q_w_3"], out_dict["scales_w_0"], out_dict["scales_w_1"],out_dict["scales_w_2"],out_dict["scales_w_3"],


# TODO: IntraNoAR decoder
# Please update forward pass here to mimic how decoder from image_model.py
# with-out framework code e.g. decoder stream.
class IntraNoAR_decoder_wrapper(IntraNoAR_wrapper):
    def __init__(self, model_path, mode="decoder", N=256, anchor_num=4, ec_thread=False, stream_part=1, inplace=False, q_in_ckpt=False):
        super().__init__(model_path=model_path, mode=mode, N=N, anchor_num=anchor_num, ec_thread=ec_thread, stream_part=stream_part, inplace=inplace, q_in_ckpt=q_in_ckpt)

    def forward(self, x, q_index, dummy_input):
        out = self.model.decompress_without_entropy_coder(x, self.q_in_ckpt, q_index, dummy_input)
        return out["x_hat"]


### DMC Wrapper

class DMC_wrapper(nn.Module):
    def __init__(self, model_path, anchor_num=4, ec_thread=False, stream_part=1, inplace=False, q_in_ckpt=False):
        super().__init__()
        self.dmc = DMC(anchor_num=anchor_num, ec_thread=ec_thread, stream_part=stream_part, inplace=inplace)
        state_dict = get_state_dict(model_path)
        self.dmc.load_state_dict(state_dict)
        self.dmc.eval()
        self.q_in_ckpt = q_in_ckpt

    def forward(self, x, ref_frame, q_index, dummy_input, frame_idx):
        dpb = { "ref_frame": ref_frame,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None,
                "ref_mv_y": None}
        encoded = self.dmc.forward_one_frame(x, dpb, q_index, dummy_input,
                                        q_in_ckpt=self.q_in_ckpt, frame_idx=frame_idx)
        # output of forward_one_frame
        return encoded["dpb"]["ref_frame"], encoded["bit"], encoded["bpp"]


class DMC_encoder_wrapper(nn.Module):
    def __init__(self, model_path, anchor_num=4, ec_thread=False, stream_part=1, inplace=False, q_in_ckpt=False):
        super().__init__()
        self.dmc = DMC(anchor_num=anchor_num, ec_thread=ec_thread, stream_part=stream_part, inplace=inplace)
        state_dict = get_state_dict(model_path)
        self.dmc.load_state_dict(state_dict)
        self.dmc.eval()
        self.q_in_ckpt = q_in_ckpt

    def forward(self, x, ref_frame, q_index, frame_idx, dummy_input):
        dpb = { "ref_frame": ref_frame,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None,
                "ref_mv_y": None}
        out = self.dmc.compress_without_entropy_coder(x, dpb, q_in_ckpt=self.q_in_ckpt,
            q_index=q_index, frame_idx=frame_idx, dummy_input=dummy_input)

        # Unfold dictionary and return elements individually
        return out["dpb"]["ref_mv_feature"], out["dpb"]["ref_y"], out["dpb"]["ref_mv_y"], out["context1"], out["context2"], out["context3"], out["z_hat"], out["y_q_w_0"], out["y_q_w_1"], out["y_q_w_2"], out["y_q_w_3"], out["scales_w_0"], out["scales_w_1"], out["scales_w_2"], out["scales_w_3"], out["mv_z_hat"], out["mv_y_q_w_0"], out["mv_y_q_w_1"], out["mv_y_q_w_2"], out["mv_y_q_w_3"], out["mv_scales_w_0"], out["mv_scales_w_1"], out["mv_scales_w_2"], out["mv_scales_w_3"]


class DMC_decoder_wrapper(nn.Module):
    def __init__(self, model_path, anchor_num=4, ec_thread=False, stream_part=1, inplace=False, q_in_ckpt=False):
        super().__init__()
        self.dmc = DMC(anchor_num=anchor_num, ec_thread=ec_thread, stream_part=stream_part, inplace=inplace)
        state_dict = get_state_dict(model_path)
        self.dmc.load_state_dict(state_dict)
        self.dmc.eval()
        self.q_in_ckpt = q_in_ckpt

    def forward(self, x, context1, context2, context3, q_index, dummy_input):
        dpb = { "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None,
                "ref_mv_y": None}
        dpb['ref_y'] = x
        input_dict = {}
        input_dict['context1'] = context1
        input_dict['context2'] = context2
        input_dict['context3'] = context3
        input_dict['dpb'] = dpb
        out = self.dmc.decompress_without_entropy_coder(input_dict, q_in_ckpt=self.q_in_ckpt,
            q_index=q_index, dummy_input=dummy_input)

        return out["dpb"]["ref_frame"], out["dpb"]["ref_feature"]
