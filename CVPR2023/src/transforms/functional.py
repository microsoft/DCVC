from typing import Tuple, Union

import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F

from torch import Tensor

YCBCR_WEIGHTS = {
    # Spec: (K_r, K_g, K_b) with K_g = 1 - K_r - K_b
    "ITU-R_BT.709": (0.2126, 0.7152, 0.0722)
}


def rgb_to_ycbcr420(rgb):
    '''
    input is 3xhxw RGB float numpy array, in the range of [0, 1]
    output is y: 1xhxw, uv: 2x(h/2)x(w/x), in the range of [0, 1]
    '''
    c, h, w = rgb.shape
    assert c == 3
    assert h % 2 == 0
    assert w % 2 == 0
    r, g, b = np.split(rgb, 3, axis=0)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5

    # to 420
    cb = np.mean(np.reshape(cb, (1, h//2, 2, w//2, 2)), axis=(-1, -3))
    cr = np.mean(np.reshape(cr, (1, h//2, 2, w//2, 2)), axis=(-1, -3))
    uv = np.concatenate((cb, cr), axis=0)

    y = np.clip(y, 0., 1.)
    uv = np.clip(uv, 0., 1.)

    return y, uv


def ycbcr420_to_rgb(y, uv, order=1):
    '''
    y is 1xhxw Y float numpy array, in the range of [0, 1]
    uv is 2x(h/2)x(w/2) UV float numpy array, in the range of [0, 1]
    order: 0 nearest neighbor, 1: binear (default)
    return value is 3xhxw RGB float numpy array, in the range of [0, 1]
    '''
    uv = scipy.ndimage.zoom(uv, (1, 2, 2), order=order)
    cb = uv[0:1, :, :]
    cr = uv[1:2, :, :]
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = np.concatenate((r, g, b), axis=0)
    rgb = np.clip(rgb, 0., 1.)
    return rgb


def ycbcr420_to_444(y, uv, order=1):
    '''
    y is 1xhxw Y float numpy array, in the range of [0, 1]
    uv is 2x(h/2)x(w/2) UV float numpy array, in the range of [0, 1]
    order: 0 nearest neighbor, 1: binear (default)
    return value is 3xhxw YCbCr float numpy array, in the range of [0, 1]
    '''
    uv = scipy.ndimage.zoom(uv, (1, 2, 2), order=order)
    yuv = np.concatenate((y, uv), axis=0)
    return yuv


def ycbcr444_to_420(yuv):
    '''
    input is 3xhxw YUV float numpy array, in the range of [0, 1]
    output is y: 1xhxw, uv: 2x(h/2)x(w/x), in the range of [0, 1]
    '''
    c, h, w = yuv.shape
    assert c == 3
    assert h % 2 == 0
    assert w % 2 == 0
    y, u, v = np.split(yuv, 3, axis=0)

    # to 420
    u = np.mean(np.reshape(u, (1, h//2, 2, w//2, 2)), axis=(-1, -3))
    v = np.mean(np.reshape(v, (1, h//2, 2, w//2, 2)), axis=(-1, -3))
    uv = np.concatenate((u, v), axis=0)

    y = np.clip(y, 0., 1.)
    uv = np.clip(uv, 0., 1.)

    return y, uv


def rgb_to_ycbcr(rgb):
    '''
    input is 3xhxw RGB float numpy array, in the range of [0, 1]
    output is yuv: 3xhxw, in the range of [0, 1]
    '''
    c, h, w = rgb.shape
    assert c == 3
    r, g, b = np.split(rgb, 3, axis=0)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5

    yuv = np.concatenate((y, cb, cr), axis=0)
    yuv = np.clip(yuv, 0., 1.)

    return yuv


def ycbcr_to_rgb(yuv):
    '''
    yuv is 3xhxw YCbCr float numpy array, in the range of [0, 1]
    return value is 3xhxw RGB float numpy array, in the range of [0, 1]
    '''
    y, cb, cr = np.split(yuv, 3, axis=0)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = np.concatenate((r, g, b), axis=0)
    rgb = np.clip(rgb, 0., 1.)
    return rgb


def _check_input_tensor(tensor: Tensor) -> None:
    if (
        not isinstance(tensor, Tensor)
        or not tensor.is_floating_point()
        or not len(tensor.size()) in (3, 4)
        or not tensor.size(-3) == 3
    ):
        raise ValueError(
            "Expected a 3D or 4D tensor with shape (Nx3xHxW) or (3xHxW) as input"
        )


def rgb2ycbcr(rgb: Tensor) -> Tensor:
    """RGB to YCbCr conversion for torch Tensor.
    Using ITU-R BT.709 coefficients.

    Args:
        rgb (torch.Tensor): 3D or 4D floating point RGB tensor

    Returns:
        ycbcr (torch.Tensor): converted tensor
    """
    _check_input_tensor(rgb)

    r, g, b = rgb.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5
    ycbcr = torch.cat((y, cb, cr), dim=-3)
    return ycbcr


def ycbcr2rgb(ycbcr: Tensor) -> Tensor:
    """YCbCr to RGB conversion for torch Tensor.
    Using ITU-R BT.709 coefficients.

    Args:
        ycbcr (torch.Tensor): 3D or 4D floating point RGB tensor

    Returns:
        rgb (torch.Tensor): converted tensor
    """
    _check_input_tensor(ycbcr)

    y, cb, cr = ycbcr.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = torch.cat((r, g, b), dim=-3)
    return rgb


def yuv_444_to_420(
    yuv: Union[Tensor, Tuple[Tensor, Tensor, Tensor]],
    mode: str = "avg_pool",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert a 444 tensor to a 420 representation.

    Args:
        yuv (torch.Tensor or (torch.Tensor, torch.Tensor, torch.Tensor)): 444
            input to be downsampled. Takes either a (Nx3xHxW) tensor or a tuple
            of 3 (Nx1xHxW) tensors.
        mode (str): algorithm used for downsampling: ``'avg_pool'``. Default
            ``'avg_pool'``

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor): Converted 420
    """
    if mode not in ("avg_pool",):
        raise ValueError(f'Invalid downsampling mode "{mode}".')

    if mode == "avg_pool":

        def _downsample(tensor):
            return F.avg_pool2d(tensor, kernel_size=2, stride=2)

    if isinstance(yuv, torch.Tensor):
        y, u, v = yuv.chunk(3, 1)
    else:
        y, u, v = yuv

    return (y, _downsample(u), _downsample(v))


def yuv_420_to_444(
    yuv: Tuple[Tensor, Tensor, Tensor],
    mode: str = "bilinear",
    return_tuple: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
    """Convert a 420 input to a 444 representation.

    Args:
        yuv (torch.Tensor, torch.Tensor, torch.Tensor): 420 input frames in
            (Nx1xHxW) format
        mode (str): algorithm used for upsampling: ``'bilinear'`` |
            ``'nearest'`` Default ``'bilinear'``
        return_tuple (bool): return input as tuple of tensors instead of a
            concatenated tensor, 3 (Nx1xHxW) tensors instead of one (Nx3xHxW)
            tensor (default: False)

    Returns:
        (torch.Tensor or (torch.Tensor, torch.Tensor, torch.Tensor)): Converted
            444
    """
    if len(yuv) != 3 or any(not isinstance(c, torch.Tensor) for c in yuv):
        raise ValueError("Expected a tuple of 3 torch tensors")

    if mode not in ("bilinear", "nearest"):
        raise ValueError(f'Invalid upsampling mode "{mode}".')

    if mode in ("bilinear", "nearest"):

        def _upsample(tensor):
            return F.interpolate(tensor, scale_factor=2, mode=mode, align_corners=False)

    y, u, v = yuv
    u, v = _upsample(u), _upsample(v)
    if return_tuple:
        return y, u, v
    return torch.cat((y, u, v), dim=1)
