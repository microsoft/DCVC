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


class VideoFolder(Dataset):
    def __init__(self, root_folder_path, patch_h, patch_w, qp_num, lambdas, frame_num=5,
                 group_of_pictures=1):
        self.root_folder_path = root_folder_path
        with open(os.path.join(self.root_folder_path, 'description.json')) as json_file:
            desc = json.load(json_file)
            self.seqs = desc['seqs']
            self.frames = desc['frames']

        self.dataset_length = len(self.seqs)
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.qp_num = qp_num
        self.lambdas = lambdas
        self.frame_num = frame_num
        self.group_of_pictures = group_of_pictures

    def __getitem__(self, index):
        seq = self.seqs[index]
        height = seq['height']
        width = seq['width']
        seq_length = seq['seq_length']
        seq_path = seq['path']

        img_indexes = []
        if self.frame_num < seq_length:
            frame_index = random.randint(0, seq_length - self.frame_num - 1)
            img_indexes = list(range(frame_index, frame_index + self.frame_num))
        else:
            increasing = True
            frame_index = 0
            while len(img_indexes) < self.frame_num:
                img_indexes.append(frame_index)
                if increasing:
                    if frame_index == seq_length - 1:
                        frame_index -= 1
                        increasing = False
                    else:
                        frame_index += 1
                else:
                    if frame_index == 0:
                        frame_index += 1
                        increasing = True
                    else:
                        frame_index -= 1

        flip = random.choice([True, False])

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

        video_data = []
        curr_img_group = []
        for img_index in img_indexes:
            img_path = os.path.join(self.root_folder_path, seq_path, self.frames[img_index])
            img = Image.open(img_path).convert('RGB')
            if flip:
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
            if len(video_data) == 0:
                video_data.append(img)
            else:
                curr_img_group.append(img)
                if len(curr_img_group) == self.group_of_pictures:
                    img = torch.cat(curr_img_group, dim=0)
                    video_data.append(img)
                    curr_img_group = []

        curr_qp = random.randint(0, self.qp_num - 1)
        curr_lambda = self.lambdas[curr_qp]
        video_data.append(torch.tensor(curr_qp, dtype=torch.int32))
        video_data.append(torch.tensor(curr_lambda, dtype=torch.float32))

        return video_data

    def __len__(self):
        return self.dataset_length

    def get_frame_num(self):
        return self.frame_num

    def get_patch_size(self):
        return self.patch_w, self.patch_h

    def set_frame_num(self, frame_num):
        self.frame_num = frame_num

    def set_patch_size(self, patch_width, patch_height):
        self.patch_w = patch_width
        self.patch_h = patch_height
