# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import struct
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor


def get_downsampled_shape(height, width, p):

    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    return int(new_h / p + 0.5), int(new_w / p + 0.5)


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def load_image(filepath: str) -> Image.Image:
    return Image.open(filepath).convert("RGB")


def img2torch(img: Image.Image) -> torch.Tensor:
    return ToTensor()(img).unsqueeze(0)


def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1).squeeze())


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def pad(x, p=2 ** 6):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )


def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )


def encode_i(height, width, y_string, z_string, output):
    with Path(output).open("wb") as f:
        y_string_length = len(y_string)
        z_string_length = len(z_string)

        write_uints(f, (height, width, y_string_length, z_string_length))
        write_bytes(f, y_string)
        write_bytes(f, z_string)


def decode_i(inputpath):
    with Path(inputpath).open("rb") as f:
        header = read_uints(f, 4)
        height = header[0]
        width = header[1]
        y_string_length = header[2]
        z_string_length = header[3]

        y_string = read_bytes(f, y_string_length)
        z_string = read_bytes(f, z_string_length)

    return height, width, y_string, z_string


def encode_p(height, width, mv_y_string, mv_z_string, y_string, z_string, output):
    with Path(output).open("wb") as f:
        mv_y_string_length = len(mv_y_string)
        mv_z_string_length = len(mv_z_string)
        y_string_length = len(y_string)
        z_string_length = len(z_string)

        write_uints(f, (height, width,
                        mv_y_string_length, mv_z_string_length,
                        y_string_length, z_string_length))
        write_bytes(f, mv_y_string)
        write_bytes(f, mv_z_string)
        write_bytes(f, y_string)
        write_bytes(f, z_string)


def decode_p(inputpath):
    with Path(inputpath).open("rb") as f:
        header = read_uints(f, 6)
        height = header[0]
        width = header[1]
        mv_y_string_length = header[2]
        mv_z_string_length = header[3]
        y_string_length = header[4]
        z_string_length = header[5]

        mv_y_string = read_bytes(f, mv_y_string_length)
        mv_z_string = read_bytes(f, mv_z_string_length)
        y_string = read_bytes(f, y_string_length)
        z_string = read_bytes(f, z_string_length)

    return height, width, mv_y_string, mv_z_string, y_string, z_string
