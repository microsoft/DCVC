# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .image_model import EVC_LL, EVC_LM, EVC_LS
from .image_model import EVC_ML, EVC_SL
from .image_model import EVC_MM, EVC_SS
from .scalable_encoder_model import Scale_EVC_SS, Scale_EVC_SL


model_architectures = {
    'EVC_LL': EVC_LL,
    'EVC_LM': EVC_LM,
    'EVC_LS': EVC_LS,

    'EVC_ML': EVC_ML,
    'EVC_SL': EVC_SL,

    'EVC_MM': EVC_MM,
    'EVC_SS': EVC_SS,

    'Scale_EVC_SS': Scale_EVC_SS,
    'Scale_EVC_SL': Scale_EVC_SL,
}


def build_model(model_name, **kwargs):
    # print(f'=> build model: {model_name}')
    if model_name in model_architectures:
        return model_architectures[model_name](**kwargs)
    else:
        raise ValueError(model_name)
