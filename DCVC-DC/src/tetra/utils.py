import numpy as np
import torch.nn as nn
import skimage

"""
Helper routines
"""
def validate(torch_outputs, coreml_outputs, output_names, psnr_threshold=60):
    for A, B, name in zip(torch_outputs, coreml_outputs, output_names):
        data_range = max(A.max() - A.min(), np.abs(A).max())
        if (A == B).all():
            psnr = np.inf
        else:
            psnr = skimage.metrics.peak_signal_noise_ratio(A, B, data_range=data_range)

        print(f"PSNR for {name}: {psnr}")
        if psnr < psnr_threshold:
            print(f"PSNR drop for {name}: {psnr} < threshold ({psnr_threshold}).\n" + f"Comparing: {A} \n {B}")

def update_torch_outputs(outputs):
    new_outputs = []
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    for each in outputs:
        out = each.detach().numpy()
        if out.shape == ():
            out = np.array([out])
        new_outputs.append(out)
    return new_outputs

