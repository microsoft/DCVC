# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import numpy as np
from PIL import Image


class PNGReader():
    def __init__(self, src_folder, width, height):
        self.src_folder = src_folder
        pngs = os.listdir(self.src_folder)
        self.width = width
        self.height = height
        if 'im1.png' in pngs:
            self.padding = 1
        elif 'im00001.png' in pngs:
            self.padding = 5
        else:
            raise ValueError('unknown image naming convention; please specify')
        self.current_frame_index = 1
        self.eof = False

    def read_one_frame(self, src_format="rgb"):
        def _none_exist_frame():
            if src_format == "rgb":
                return None
            return None, None, None
        if self.eof:
            return _none_exist_frame()

        png_path = os.path.join(self.src_folder,
                                f"im{str(self.current_frame_index).zfill(self.padding)}.png"
                                )
        if not os.path.exists(png_path):
            self.eof = True
            return _none_exist_frame()

        rgb = Image.open(png_path).convert('RGB')
        rgb = np.asarray(rgb).astype('float32').transpose(2, 0, 1)
        rgb = rgb / 255.
        _, height, width = rgb.shape
        assert height == self.height
        assert width == self.width

        self.current_frame_index += 1
        return rgb

    def close(self):
        self.current_frame_index = 1
