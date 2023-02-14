# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from src.models import build_model
from ptflops import get_model_complexity_info


class IntraCodec(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        with torch.no_grad():
            result = self.model.forward(x, q_scale=1.0)
        return result


def print_model():
    net = build_model("EVC_SS")
    Codec = IntraCodec
    model = Codec(net)
    img_size = (3, 1920, 1088)

    macs, params = get_model_complexity_info(model, img_size, as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    print(f" macs {macs}  params {params}")


if __name__ == "__main__":
    print_model()
