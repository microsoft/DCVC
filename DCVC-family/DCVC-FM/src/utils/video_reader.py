# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import numpy as np
from PIL import Image
from ..transforms.functional import rgb_to_ycbcr420, ycbcr420_to_rgb, ycbcr444_to_rgb, \
    ycbcr444_to_420, rgb_to_ycbcr444


class VideoReader():
    def __init__(self, src_path, width, height):
        self.src_path = src_path
        self.width = width
        self.height = height
        self.eof = False

    def read_one_frame(self, dst_format='rgb'):
        '''
        y is 1xhxw Y float numpy array, in the range of [0, 1]
        uv is 2x(h/2)x(w/2) UV float numpy array, in the range of [0, 1]
        rgb is 3xhxw float numpy array, in the range of [0, 1]
        '''
        raise NotImplementedError

    @staticmethod
    def _none_exist_frame(dst_format):
        if dst_format == "420":
            return None, None
        assert dst_format == "rgb"
        return None

    @staticmethod
    def _get_dst_format(rgb=None, y=None, uv=None, src_format='rgb', dst_format='rgb'):
        if dst_format == 'rgb':
            if src_format == '420':
                rgb = ycbcr420_to_rgb(y, uv, order=1)
            elif src_format == '444':
                rgb = ycbcr444_to_rgb(y, uv)
            return rgb
        elif dst_format == '420':
            if src_format == 'rgb':
                y, uv = rgb_to_ycbcr420(rgb)
            elif src_format == '444':
                y, uv = ycbcr444_to_420(np.concatenate((y, uv), axis=0))
            return y, uv
        elif dst_format == '444':
            if src_format == 'rgb':
                y, uv = rgb_to_ycbcr444(rgb)
            elif src_format == '420':
                y, uv = ycbcr444_to_420(y, uv)
            return y, uv
        assert False


class PNGReader(VideoReader):
    def __init__(self, src_path, width, height, start_num=1):
        super().__init__(src_path, width, height)

        pngs = os.listdir(self.src_path)
        if 'im1.png' in pngs:
            self.padding = 1
        elif 'im00001.png' in pngs:
            self.padding = 5
        else:
            raise ValueError('unknown image naming convention; please specify')
        self.current_frame_index = start_num

    def read_one_frame(self, dst_format="rgb"):
        if self.eof:
            return self._none_exist_frame(dst_format)

        png_path = os.path.join(self.src_path,
                                f"im{str(self.current_frame_index).zfill(self.padding)}.png"
                                )
        if not os.path.exists(png_path):
            self.eof = True
            return self._none_exist_frame(dst_format)

        rgb = Image.open(png_path).convert('RGB')
        rgb = np.asarray(rgb).astype('float32').transpose(2, 0, 1)
        rgb = rgb / 255.
        _, height, width = rgb.shape
        assert height == self.height
        assert width == self.width

        self.current_frame_index += 1
        return self._get_dst_format(rgb=rgb, src_format='rgb', dst_format=dst_format)

    def close(self):
        self.current_frame_index = 1


class RGBReader(VideoReader):
    def __init__(self, src_path, width, height, src_format='rgb', bit_depth=8):
        super().__init__(src_path, width, height)

        self.src_format = src_format
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
        self.file = open(src_path, "rb")
        # pylint: enable=R1732

    def read_one_frame(self, dst_format="420"):
        if self.eof:
            return self._none_exist_frame(dst_format)
        rgb = self.file.read(self.rgb_size)
        if not rgb:
            self.eof = True
            return self._none_exist_frame(dst_format)
        rgb = np.frombuffer(rgb, dtype=self.dtype).copy().reshape(3, self.height, self.width)
        rgb = rgb.astype(np.float32) / self.max_val

        return self._get_dst_format(rgb=rgb, src_format='rgb', dst_format=dst_format)

    def close(self):
        self.file.close()


class YUVReader(VideoReader):
    def __init__(self, src_path, width, height, src_format='420', bit_depth=8, skip_frame=0):
        super().__init__(src_path, width, height)
        if not src_path.endswith('.yuv'):
            src_path = src_path + '.yuv'
            self.src_path = src_path

        self.src_format = src_format
        self.y_size = width * height
        self.uv_size = self.y_size * 2
        self.uv_width = width
        self.uv_height = height
        self.src_format = '444'
        if src_format == '420':
            self.src_format = '420'
            self.uv_size = width * height // 2
            self.uv_width = width // 2
            self.uv_height = height // 2
        self.dtype = np.uint8
        self.max_val = 255
        if bit_depth > 8 and bit_depth <= 16:
            self.y_size = self.y_size * 2
            self.uv_size = self.uv_size * 2
            self.dtype = np.uint16
            self.max_val = (1 << bit_depth) - 1
        else:
            assert bit_depth == 8
        # pylint: disable=R1732
        self.file = open(src_path, "rb")
        # pylint: enable=R1732
        skipped_frame = 0
        while not self.eof and skipped_frame < skip_frame:
            y = self.file.read(self.y_size)
            uv = self.file.read(self.uv_size)
            if not y or not uv:
                self.eof = True
            skipped_frame += 1

    def read_one_frame(self, dst_format="420"):
        if self.eof:
            return self._none_exist_frame(dst_format)
        y = self.file.read(self.y_size)
        uv = self.file.read(self.uv_size)
        if not y or not uv:
            self.eof = True
            return self._none_exist_frame(dst_format)
        y = np.frombuffer(y, dtype=self.dtype).copy().reshape(1, self.height, self.width)
        uv = np.frombuffer(uv, dtype=self.dtype).copy().reshape(2, self.uv_height, self.uv_width)
        y = y.astype(np.float32) / self.max_val
        uv = uv.astype(np.float32) / self.max_val

        return self._get_dst_format(y=y, uv=uv, src_format=self.src_format, dst_format=dst_format)

    def close(self):
        self.file.close()
