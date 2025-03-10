import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F


YCBCR_WEIGHTS = {
    # Spec: (K_r, K_g, K_b) with K_g = 1 - K_r - K_b
    "ITU-R_BT.709": (0.2126, 0.7152, 0.0722)
}


def ycbcr420_to_444_np(y, uv, order=0, separate=False):
    '''
    y is 1xhxw Y float numpy array
    uv is 2x(h/2)x(w/2) UV float numpy array
    order: 0 nearest neighbor (default), 1: binear
    return value is 3xhxw YCbCr float numpy array
    '''
    uv = scipy.ndimage.zoom(uv, (1, 2, 2), order=order)
    if separate:
        return y, uv
    yuv = np.concatenate((y, uv), axis=0)
    return yuv


def rgb2ycbcr(rgb, is_bgr=False):
    if is_bgr:
        b, g, r = rgb.chunk(3, -3)
    else:
        r, g, b = rgb.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5
    ycbcr = torch.cat((y, cb, cr), dim=-3)
    ycbcr = torch.clamp(ycbcr, 0., 1.)
    return ycbcr


def ycbcr2rgb(ycbcr, is_bgr=False, clamp=True):
    y, cb, cr = ycbcr.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    if is_bgr:
        rgb = torch.cat((b, g, r), dim=-3)
    else:
        rgb = torch.cat((r, g, b), dim=-3)
    if clamp:
        rgb = torch.clamp(rgb, 0., 1.)
    return rgb


def yuv_444_to_420(yuv):
    def _downsample(tensor):
        return F.avg_pool2d(tensor, kernel_size=2, stride=2)

    y = yuv[:, :1, :, :]
    uv = yuv[:, 1:, :, :]

    return y, _downsample(uv)
