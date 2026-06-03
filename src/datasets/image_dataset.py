# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import numpy as np
import os
import random
import torch

from PIL import Image
from torch.utils.data import Dataset

from ..utils.transforms import rgb2ycbcr_np


class ImageFolder(Dataset):
    def __init__(self, root_folder_path, patch_h, patch_w, qp_num, lambdas):
        self.root_folder_path = root_folder_path
        with open(os.path.join(root_folder_path, 'description.json')) as json_file:
            self.dataset = json.load(json_file)

        self.dataset_length = len(self.dataset)
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.qp_num = qp_num
        self.lambdas = lambdas

    def __getitem__(self, index):
        image_path = os.path.join(self.root_folder_path, self.dataset[index])
        img = Image.open(image_path).convert('RGB')
        width, height = img.size

        pad_height = self.patch_h - height
        pad_width = self.patch_w - width
        pad_height = max(0, pad_height)
        pad_width = max(0, pad_width)
        pad_size = ((pad_height // 2, pad_height - pad_height // 2),
                    (pad_width // 2, pad_width - pad_width // 2),
                    (0, 0),)
        padded_height = height + pad_height
        padded_width = width + pad_width
        y = random.randint(0, padded_height - self.patch_h)
        x = random.randint(0, padded_width - self.patch_w)

        if random.choice([True, False]):
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        img = np.array(img).astype(np.uint8)
        img = np.pad(img, pad_size, mode='constant')
        img = img[y:y+self.patch_h, x:x+self.patch_w, :]

        img = img.astype(np.float32) / 255.0
        img = rgb2ycbcr_np(img)
        img = img - 0.5
        img = torch.as_tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1).contiguous()
        # img in [3, H, W]

        curr_qp = random.randint(0, self.qp_num - 1)
        curr_lambda = self.lambdas[curr_qp]
        curr_qp = torch.tensor(curr_qp, dtype=torch.int32)
        curr_lambda = torch.tensor(curr_lambda, dtype=torch.float32)
        return [img, curr_qp, curr_lambda]

    def __len__(self):
        return self.dataset_length

    def get_patch_size(self):
        return self.patch_w, self.patch_h

    def set_patch_size(self, patch_width, patch_height):
        self.patch_w = patch_width
        self.patch_h = patch_height
