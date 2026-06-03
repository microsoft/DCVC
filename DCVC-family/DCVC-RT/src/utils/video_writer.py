# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

from PIL import Image


class PNGWriter():
    def __init__(self, dst_path, width, height):
        self.dst_path = dst_path
        self.width = width
        self.height = height
        self.padding = 5
        self.current_frame_index = 1
        os.makedirs(dst_path, exist_ok=True)

    def write_one_frame(self, rgb):
        # rgb: 3xhxw uint8 numpy array
        rgb = rgb.transpose(1, 2, 0)

        png_path = os.path.join(self.dst_path,
                                f"im{str(self.current_frame_index).zfill(self.padding)}.png"
                                )
        Image.fromarray(rgb).save(png_path)

        self.current_frame_index += 1

    def close(self):
        self.current_frame_index = 1


class YUV420Writer():
    def __init__(self, dst_path, width, height):
        if not dst_path.endswith('.yuv'):
            dst_path = dst_path + '/out.yuv'
        self.dst_path = dst_path
        self.width = width
        self.height = height

        # pylint: disable=R1732
        self.file = open(dst_path, "wb")
        # pylint: enable=R1732

    def write_one_frame(self, y, uv):
        # y: 1xhxw uint8 numpy array
        # uv: 2x(h/2)x(w/2) uint8 numpy array
        self.file.write(y.tobytes())
        self.file.write(uv.tobytes())

    def close(self):
        self.file.close()
