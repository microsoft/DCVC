from ..models.image_model import IntraNoAR
from ..models.video_model import DMC
from ..utils.stream_helper import get_state_dict

import torch.nn as nn

## IntraNoAr wrapper

class IntraNoAR_wrapper(nn.Module):
    def __init__(self, model_path, mode="forward", N=256, anchor_num=4, ec_thread=False, stream_part=1, inplace=False, q_in_ckpt=False, q_index=0):
        super().__init__()
        self.model = IntraNoAR(N, anchor_num, ec_thread, stream_part, inplace)
        state_dict = get_state_dict(model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        assert mode in { "forward", "encoder", "decoder"}
        self.mode = mode
        self.torch_output_order = None
        self.q_in_ckpt = q_in_ckpt
        self.q_index = q_index

    def forward(self, x):
        out_dict = self.model(x, q_in_ckpt=self.q_in_ckpt, q_index=self.q_index)
        return out_dict["x_hat"], out_dict["bit"], out_dict["bpp"], out_dict["bpp_y"], out_dict["bpp_z"]


# TODO: IntraNoAR encoder
# Please update forward pass here to mimic how encoder from image_model.py
# with-out framework code e.g. encoder stream.
class IntraNoAR_encoder_wrapper(IntraNoAR_wrapper):
    def __init__(self, model_path):
        super().__init__(model_path=model_path, mode="encoder")

    def forward(self, x, q_in_ckpt=False, q_index=0):
        # TODO: add remaining pieces of encoder
        # Reference: image_model.py::IntraNoAR::forward
        curr_q_enc, _ = self.model.get_q_for_inference(q_in_ckpt=q_in_ckpt, q_index=q_index)
        y = self.model.enc(x, curr_q_enc)
        # NOTE: Skipping pad for simplicity.
        # e.g. if input is 720:
        # y is of shape (1, 256, 45, 80)
        # padding is (0, tensor(0), 0, tensor(-3))
        # y_pad becomes torch.Size([1, 256, 48, 80])
        # We need to store how much was padded for decoder
        # to get back to original resolution
        # y_pad, _ = self.model.pad_for_y(y)
        z = self.model.hyper_enc(y)
        z_hat = self.model.quant(z)
        return z_hat

# TODO: IntraNoAR decoder
# Please update forward pass here to mimic how decoder from image_model.py
# with-out framework code e.g. decoder stream.
class IntraNoAR_decoder_wrapper(IntraNoAR_wrapper):
    def __init__(self, model_path):
        super().__init__(model_path=model_path, mode="decoder")

    def forward(self, x, q_in_ckpt=False, q_index=0):
        # TODO: add remaining pieces of decoder
        _, curr_q_dec = self.model.get_q_for_inference(q_in_ckpt=q_in_ckpt, q_index=q_index)
        y_hat = self.model.hyper_dec(x)
        # params = self.model.y_prior_fusion(y_hat)
        # NOTE: refer to notes from encoder for Skipping pad for simplicity
        # params = self.model.slice_to_y(params, slice_shape)
        # TODO: Update and use decompression correctly.
        # _, y_q, y_hat, scales_hat = self.model.forward_four_part_prior(
        #     y_hat, params, self.model.y_spatial_prior_adaptor_1, self.model.y_spatial_prior_adaptor_2,
        #     self.model.y_spatial_prior_adaptor_3, self.model.y_spatial_prior)
        # y_hat = self.model.decompress_four_part_prior_with_y(y_hat, y_hat,
        #                                         self.model.y_spatial_prior_adaptor_1,
        #                                         self.model.y_spatial_prior_adaptor_2,
        #                                         self.model.y_spatial_prior_adaptor_3,
        #                                         self.model.y_spatial_prior)
        y_hat = self.model.dec(y_hat, curr_q_dec)
        y_hat = self.model.refine(y_hat)
        y_hat = y_hat.clamp_(0, 1)
        return y_hat


### DMC Wrapper

class DMC_wrapper(nn.Module):
    def __init__(self, model_path, anchor_num=4, ec_thread=False, stream_part=1, inplace=False, q_in_ckpt=False, q_index=0, frame_idx=0):
        super().__init__()
        self.dmc = DMC(anchor_num=anchor_num, ec_thread=ec_thread, stream_part=stream_part, inplace=inplace)
        state_dict = get_state_dict(model_path)
        self.dmc.load_state_dict(state_dict)
        self.dmc.eval()
        self.q_in_ckpt = q_in_ckpt
        self.q_index = q_index
        self.frame_idx = frame_idx

    def forward(self, x, ref_frame):
        dpb = { "ref_frame": ref_frame,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None,
                "ref_mv_y": None}
        encoded = self.dmc.forward_one_frame(x, dpb, q_in_ckpt=self.q_in_ckpt, q_index=self.q_index,
                                         frame_idx=self.frame_idx)
        # output of forward_one_frame
        # "bpp_mv_y": bpp_mv_y,
        #         "bpp_mv_z": bpp_mv_z,
        #         "bpp_y": bpp_y,
        #         "bpp_z": bpp_z,
        #         "bpp": bpp,
        #         "dpb": {
        #             "ref_frame": x_hat,
        #             "ref_feature": feature,
        #             "ref_mv_feature": mv_feature,
        #             "ref_y": y_hat,
        #             "ref_mv_y": mv_y_hat,
        #         },
        #         "bit": bit,
        #         "bit_y": bit_y,
        #         "bit_z": bit_z,
        #         "bit_mv_y": bit_mv_y,
        #         "bit_mv_z": bit_mv_z,
        #         }
        return encoded["dpb"]["ref_frame"], encoded["bit"], encoded["bpp"]