# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import numpy as np
from PIL import Image


class PNGReader():
    def __init__(self, filepath):
        self.filepath = filepath
        self.eof = False

    def read_one_frame(self, src_format="rgb"):
        if self.eof:
            return None

        png_path = self.filepath
        if not os.path.exists(png_path):
            self.eof = True
            return None

        rgb = Image.open(png_path).convert('RGB')
        rgb = np.asarray(rgb).astype('float32').transpose(2, 0, 1)
        rgb = rgb / 255.
        return rgb
