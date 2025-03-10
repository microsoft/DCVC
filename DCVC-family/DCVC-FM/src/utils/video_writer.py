# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import numpy as np
from PIL import Image
from ..transforms.functional import ycbcr420_to_rgb, rgb_to_ycbcr420, ycbcr444_to_rgb, \
    rgb_to_ycbcr444, ycbcr420_to_444, ycbcr444_to_420


class VideoWriter():
    def __init__(self, dst_path, width, height):
        self.dst_path = dst_path
        self.width = width
        self.height = height

    def write_one_frame(self, rgb=None, y=None, uv=None, src_format="rgb"):
        '''
        y is 1xhxw Y float numpy array, in the range of [0, 1]
        uv is 2x(h/2)x(w/2) UV float numpy array, in the range of [0, 1]
        rgb is 3xhxw float numpy array, in the range of [0, 1]
        '''
        raise NotImplementedError


class PNGWriter(VideoWriter):
    def __init__(self, dst_path, width, height):
        super().__init__(dst_path, width, height)
        self.padding = 5
        self.current_frame_index = 1
        os.makedirs(dst_path, exist_ok=True)

    def write_one_frame(self, rgb=None, y=None, uv=None, src_format="rgb"):
        if src_format == "420":
            rgb = ycbcr420_to_rgb(y, uv, order=1)
        elif src_format == "444":
            rgb = ycbcr444_to_rgb(y, uv)
        rgb = rgb.transpose(1, 2, 0)

        png_path = os.path.join(self.dst_path,
                                f"im{str(self.current_frame_index).zfill(self.padding)}.png"
                                )
        img = np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)
        Image.fromarray(img).save(png_path)

        self.current_frame_index += 1

    def close(self):
        self.current_frame_index = 1


class RGBWriter(VideoWriter):
    def __init__(self, dst_path, width, height, dst_format='rgb', bit_depth=8):
        super().__init__(dst_path, width, height)

        self.dst_format = dst_format
        self.bit_depth = bit_depth
        self.rgb_size = width * height * 3
        self.dtype = np.uint8
        self.max_val = 255
        if bit_depth > 8 and bit_depth <= 16:
            self.rgb_size = self.rgb_size * 2
            self.dtype = np.uint16
            self.max_val = (1 << bit_depth) - 1
        else:
            assert bit_depth == 8
        # pylint: disable=R1732
        self.file = open(dst_path, "wb")
        # pylint: enable=R1732

    def write_one_frame(self, rgb=None, y=None, uv=None, src_format="rgb"):
        if src_format == '420':
            rgb = ycbcr420_to_rgb(y, uv, order=1)
        elif src_format == '444':
            rgb = ycbcr444_to_rgb(y, uv)
        rgb = np.clip(np.rint(rgb * self.max_val), 0, self.max_val).astype(self.dtype)

        self.file.write(rgb.tobytes())

    def close(self):
        self.file.close()


class YUVWriter(VideoWriter):
    def __init__(self, dst_path, width, height, dst_format='420', bit_depth=8):
        super().__init__(dst_path, width, height)
        if not dst_path.endswith('.yuv'):
            dst_path = dst_path + '/out.yuv'
            self.dst_path = dst_path

        self.dst_format = dst_format
        self.y_size = width * height
        self.uv_size = width * height
        if dst_format == '420':
            self.uv_size = width * height // 2
        self.bit_depth = bit_depth
        self.dtype = np.uint8
        self.max_val = 255
        if bit_depth > 8 and bit_depth <= 16:
            self.y_size = self.y_size * 2
            self.uv_size = self.uv_size * 2
            self.dtype = np.uint16
            self.max_val = (1 << bit_depth) - 1
        else:
            assert bit_depth == 8
        self.eof = False
        # pylint: disable=R1732
        self.file = open(dst_path, "wb")
        # pylint: enable=R1732

    def write_one_frame(self, rgb=None, y=None, uv=None, src_format="420"):
        if src_format == 'rgb':
            if self.dst_format == '420':
                y, uv = rgb_to_ycbcr420(rgb)
            elif self.dst_format == '444':
                y, uv = rgb_to_ycbcr444(rgb)
            else:
                assert False
        elif src_format == '420' and self.dst_format == '444':
            y, uv = ycbcr420_to_444(y, uv, separate=True)
        elif src_format == '444' and self.dst_format == '420':
            y, uv = ycbcr444_to_420(y, uv)
        y = np.clip(np.rint(y * self.max_val), 0, self.max_val).astype(self.dtype)
        uv = np.clip(np.rint(uv * self.max_val), 0, self.max_val).astype(self.dtype)

        self.file.write(y.tobytes())
        self.file.write(uv.tobytes())

    def close(self):
        self.file.close()
