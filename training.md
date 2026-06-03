# Training DCVC-UF

This document describes how to train DCVC-UF and DCVC-UF-Intra from scratch, including the dataset format, launcher scripts, and the multi-stage training schedule.

## Train dataset

Training uses RGB PNG images. The dataset loaders are implemented in `src/datasets/image_dataset.py` (`ImageFolder`) and `src/datasets/video_dataset.py` (`VideoFolder`).

### Image dataset

Organize a collection of PNG images under a root folder and provide a `description.json` that lists the relative path of each image:

```
/path/to/image_dataset/
├── description.json
├── img_000001.png
├── img_000002.png
└── ...
```

`description.json` is a flat JSON array of relative image paths:
```json
[
    "img_000001.png",
    "img_000002.png",
    "..."
]
```

### Video dataset

Organize video sequences as folders of numbered PNG frames under a root folder and provide a `description.json` that describes the sequences:

```
/path/to/video_dataset/
├── description.json
├── sequence_001/
│   ├── im00001.png
│   ├── im00002.png
│   └── ...
├── sequence_002/
│   ├── im00001.png
│   └── ...
└── ...
```

`description.json` contains two fields — `seqs` (a list of sequence metadata) and `frames` (a list of frame filenames):
```json
{
    "seqs": [
        {
            "path": "sequence_001",
            "width": 1920,
            "height": 1080,
            "seq_length": 100
        },
        {
            "path": "sequence_002",
            "width": 1280,
            "height": 720,
            "seq_length": 200
        }
    ],
    "frames": [
        "im00001.png",
        "im00002.png",
        "im00003.png",
        "..."
    ]
}
```

The `frames` list is shared across all sequences (i.e., all sequences use the same frame naming scheme). Each entry in `seqs` specifies the subdirectory path, resolution, and number of frames for that sequence.

## Train the image model (DCVC-UF-Intra)

Create a `train_image.sh` file with the contents below, edit the user-configurable variables, then run:

```bash
bash train_image.sh
```

The script trains for 105 epochs with an adaptive learning rate schedule (1e-6 to 2e-4) and progressive patch sizes (256x256 → 512x512).

**Training dataset:** We used subsets 0, 1, and 2 of the [Open Images](https://github.com/cvdfoundation/open-images-dataset) training set.

### `train_image.sh`

```bash
#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Train the DMCI image model from scratch.
# 1 stage: 105 epochs, 256x256 patches for first 90 epochs, then 512x512.

set -e

# ========== User-configurable variables ==========
TRAIN_DATASET=/path/to/image_dataset
SAVE_DIR=/path/to/output/image_model
LAMBDAS="10 2048"
# ==================================================

python train_image.py \
    --train_dataset ${TRAIN_DATASET} \
    --save_dir ${SAVE_DIR} \
    --lambdas ${LAMBDAS} \
    --batch_size 16 \
    -n 4 \
    -e 105
```

## Train the video model (DCVC-UF)

Each variant uses its own launcher script. Create the launcher file shown below for the variant(s) you want to train, edit the user-configurable variables (dataset paths, pretrained image model path, save directory), then run:

```bash
bash train_video_htl.sh   # HT-L variant
bash train_video_hts.sh   # HT-S variant
bash train_video_ld.sh    # LD variant
```

Training requires a pretrained image model (DCVC-UF-Intra). Each script runs a multi-stage pipeline that progressively increases the sequence length and patch size. Longer training sequences enable the model to learn more comprehensive temporal dependencies through cross-chunk feature propagation, which is a key advantage of the chunk-based framework.

- **HT-L/HT-S**: 4 stages, 99 epochs total, with cascading loss from stage 1
- **LD**: 4 stages, 136 epochs total, with cascading loss from stage 1

**Training datasets:** We used two video datasets for training:
1. The septuplet subset of [Vimeo-90k](http://toflow.csail.mit.edu/index.html).
2. We also process the [original videos](http://data.csail.mit.edu/tofu/dataset/original_video_list.txt) of [Vimeo-90k](http://toflow.csail.mit.edu/index.html) to generate long sequences.

### `train_video_htl.sh`

```bash
#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Train the DMC-HTL video model (high throughput, frame_delay=8).
# 4 stages:
#   stage0: 45 epochs, non-cascaded, 256x256
#   stage1: 10 epochs, cascaded, 256x256
#   stage2: 40 epochs, cascaded, 512x512
#   stage3:  4 epochs, cascaded, 512x512

set -e

# ========== User-configurable variables ==========
TRAIN_DATASET_STAGE0=/path/to/video_dataset_stage0
TRAIN_DATASET_STAGE1=/path/to/video_dataset_stage1
TRAIN_DATASET_STAGE2=/path/to/video_dataset_stage2
TRAIN_DATASET_STAGE3=/path/to/video_dataset_stage3

MODEL_PATH_I=/path/to/image_model.pth.tar
SAVE_DIR=/path/to/output/video_model_htl
LAMBDAS="1 768"
# ==================================================

MODEL_STRUCTURE=htl

# stage0: base training
python train_video.py \
    --model_structure ${MODEL_STRUCTURE} \
    --model_path_i ${MODEL_PATH_I} \
    --train_dataset ${TRAIN_DATASET_STAGE0} \
    --save_dir ${SAVE_DIR}/s0 \
    --lambdas ${LAMBDAS} \
    --training_scheduling stage0 \
    --batch_size 16 \
    -n 4 \
    -e 45

# stage1: cascaded finetuning at 256x256
python train_video.py \
    --model_structure ${MODEL_STRUCTURE} \
    --model_path_i ${MODEL_PATH_I} \
    --train_dataset ${TRAIN_DATASET_STAGE1} \
    --save_dir ${SAVE_DIR}/s1 \
    --lambdas ${LAMBDAS} \
    --training_scheduling stage1 \
    --pretrain_path ${SAVE_DIR}/s0/ckpt.pth.tar \
    --batch_size 8 \
    -n 4 \
    -e 10

# stage2: cascaded finetuning at 512x512
python train_video.py \
    --model_structure ${MODEL_STRUCTURE} \
    --model_path_i ${MODEL_PATH_I} \
    --train_dataset ${TRAIN_DATASET_STAGE2} \
    --save_dir ${SAVE_DIR}/s2 \
    --lambdas ${LAMBDAS} \
    --training_scheduling stage2 \
    --pretrain_path ${SAVE_DIR}/s1/ckpt.pth.tar \
    --batch_size 8 \
    -n 8 \
    -e 40

# stage3: cascaded finetuning at 512x512 with longer sequences
python train_video.py \
    --model_structure ${MODEL_STRUCTURE} \
    --model_path_i ${MODEL_PATH_I} \
    --train_dataset ${TRAIN_DATASET_STAGE3} \
    --save_dir ${SAVE_DIR}/s3 \
    --lambdas ${LAMBDAS} \
    --training_scheduling stage3 \
    --pretrain_path ${SAVE_DIR}/s2/ckpt.pth.tar \
    --batch_size 16 \
    -n 8 \
    -e 4
```

### `train_video_hts.sh`

```bash
#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Train the DMC-HTS video model (high throughput, frame_delay=8).
# 4 stages:
#   stage0: 45 epochs, non-cascaded, 256x256
#   stage1: 10 epochs, cascaded, 256x256
#   stage2: 40 epochs, cascaded, 512x512
#   stage3:  4 epochs, cascaded, 512x512

set -e

# ========== User-configurable variables ==========
TRAIN_DATASET_STAGE0=/path/to/video_dataset_stage0
TRAIN_DATASET_STAGE1=/path/to/video_dataset_stage1
TRAIN_DATASET_STAGE2=/path/to/video_dataset_stage2
TRAIN_DATASET_STAGE3=/path/to/video_dataset_stage3

MODEL_PATH_I=/path/to/image_model.pth.tar
SAVE_DIR=/path/to/output/video_model_hts
LAMBDAS="1 768"
# ==================================================

MODEL_STRUCTURE=hts

# stage0: base training
python train_video.py \
    --model_structure ${MODEL_STRUCTURE} \
    --model_path_i ${MODEL_PATH_I} \
    --train_dataset ${TRAIN_DATASET_STAGE0} \
    --save_dir ${SAVE_DIR}/s0 \
    --lambdas ${LAMBDAS} \
    --training_scheduling stage0 \
    --batch_size 16 \
    -n 4 \
    -e 45

# stage1: cascaded finetuning at 256x256
python train_video.py \
    --model_structure ${MODEL_STRUCTURE} \
    --model_path_i ${MODEL_PATH_I} \
    --train_dataset ${TRAIN_DATASET_STAGE1} \
    --save_dir ${SAVE_DIR}/s1 \
    --lambdas ${LAMBDAS} \
    --training_scheduling stage1 \
    --pretrain_path ${SAVE_DIR}/s0/ckpt.pth.tar \
    --batch_size 8 \
    -n 4 \
    -e 10

# stage2: cascaded finetuning at 512x512
python train_video.py \
    --model_structure ${MODEL_STRUCTURE} \
    --model_path_i ${MODEL_PATH_I} \
    --train_dataset ${TRAIN_DATASET_STAGE2} \
    --save_dir ${SAVE_DIR}/s2 \
    --lambdas ${LAMBDAS} \
    --training_scheduling stage2 \
    --pretrain_path ${SAVE_DIR}/s1/ckpt.pth.tar \
    --batch_size 8 \
    -n 8 \
    -e 40

# stage3: cascaded finetuning at 512x512 with longer sequences
python train_video.py \
    --model_structure ${MODEL_STRUCTURE} \
    --model_path_i ${MODEL_PATH_I} \
    --train_dataset ${TRAIN_DATASET_STAGE3} \
    --save_dir ${SAVE_DIR}/s3 \
    --lambdas ${LAMBDAS} \
    --training_scheduling stage3 \
    --pretrain_path ${SAVE_DIR}/s2/ckpt.pth.tar \
    --batch_size 16 \
    -n 8 \
    -e 4
```

### `train_video_ld.sh`

```bash
#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Train the DMC-LD video model (low delay, frame_delay=1).
# 4 stages:
#   stage0: 55 epochs, non-cascaded, 256x256
#   stage1: 37 epochs, cascaded, 256x256
#   stage2: 40 epochs, cascaded, 512x512
#   stage3:  4 epochs, cascaded, 512x512

set -e

# ========== User-configurable variables ==========
TRAIN_DATASET_STAGE0=/path/to/video_dataset_stage0
TRAIN_DATASET_STAGE1=/path/to/video_dataset_stage1
TRAIN_DATASET_STAGE2=/path/to/video_dataset_stage2
TRAIN_DATASET_STAGE3=/path/to/video_dataset_stage3

MODEL_PATH_I=/path/to/image_model.pth.tar
SAVE_DIR=/path/to/output/video_model_ld
LAMBDAS="1 768"
# ==================================================

MODEL_STRUCTURE=ld

# stage0: base training
python train_video.py \
    --model_structure ${MODEL_STRUCTURE} \
    --model_path_i ${MODEL_PATH_I} \
    --train_dataset ${TRAIN_DATASET_STAGE0} \
    --save_dir ${SAVE_DIR}/s0 \
    --lambdas ${LAMBDAS} \
    --training_scheduling stage0 \
    --batch_size 16 \
    -n 4 \
    -e 55

# stage1: cascaded finetuning at 256x256
python train_video.py \
    --model_structure ${MODEL_STRUCTURE} \
    --model_path_i ${MODEL_PATH_I} \
    --train_dataset ${TRAIN_DATASET_STAGE1} \
    --save_dir ${SAVE_DIR}/s1 \
    --lambdas ${LAMBDAS} \
    --training_scheduling stage1 \
    --pretrain_path ${SAVE_DIR}/s0/ckpt.pth.tar \
    --batch_size 8 \
    -n 4 \
    -e 37

# stage2: cascaded finetuning at 512x512
python train_video.py \
    --model_structure ${MODEL_STRUCTURE} \
    --model_path_i ${MODEL_PATH_I} \
    --train_dataset ${TRAIN_DATASET_STAGE2} \
    --save_dir ${SAVE_DIR}/s2 \
    --lambdas ${LAMBDAS} \
    --training_scheduling stage2 \
    --pretrain_path ${SAVE_DIR}/s1/ckpt.pth.tar \
    --batch_size 8 \
    -n 8 \
    -e 40

# stage3: cascaded finetuning at 512x512 with longer sequences
python train_video.py \
    --model_structure ${MODEL_STRUCTURE} \
    --model_path_i ${MODEL_PATH_I} \
    --train_dataset ${TRAIN_DATASET_STAGE3} \
    --save_dir ${SAVE_DIR}/s3 \
    --lambdas ${LAMBDAS} \
    --training_scheduling stage3 \
    --pretrain_path ${SAVE_DIR}/s2/ckpt.pth.tar \
    --batch_size 4 \
    -n 8 \
    -e 4
```
