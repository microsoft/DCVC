# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os


def test_one_model(i_frame_model, checkpoint):
    root_folder = "output"
    output_json_path = f"{root_folder}/output_json/arch:{i_frame_model}_ckpt:{checkpoint}.json"
    image_model = f"checkpoints/{checkpoint}"

    test_cfg = 'local_kodak.json'

    command_line = (" python test_image.py "
                    f" --i_frame_model {i_frame_model}"
                    f" --i_frame_model_path {image_model}"
                    f" --test_config ./test_cfg/{test_cfg}"
                    " --cuda 1 -w 1 --rate_num 4"
                    " --write_stream 0 --ec_thread 1"
                    " --verbose 1"
                    # " --save_decoded_frame True"
                    f" --output_path {output_json_path}")

    print(command_line)
    os.system(command_line)


def main():
    # i_frame_model = "EVC_LL"
    # checkpoint = 'EVC_LL.pth.tar'

    # i_frame_model = "EVC_ML"
    # checkpoint = 'EVC_ML_MD.pth.tar'

    # i_frame_model = "EVC_SL"
    # checkpoint = 'EVC_SL_MD.pth.tar'

    # i_frame_model = "EVC_LM"
    # checkpoint = 'EVC_LM_MD.pth.tar'

    # i_frame_model = "EVC_LS"
    # checkpoint = 'EVC_LS_MD.pth.tar'

    # i_frame_model = "EVC_MM"
    # checkpoint = 'EVC_MM_MD.pth.tar'

    i_frame_model = "EVC_SS"
    checkpoint = 'EVC_SS_MD.pth.tar'

    # i_frame_model = "Scale_EVC_SL"
    # checkpoint = 'Scale_EVC_SL_MDRRL.pth.tar'

    # i_frame_model = "Scale_EVC_SS"
    # checkpoint = 'Scale_EVC_SS_MDRRL.pth.tar'
    test_one_model(i_frame_model, checkpoint)


# latency on kodak
"""
CUDA_VISIBLE_DEVICES=0 python test_all_image.py 2>&1 | tee "log.txt"
cat log.txt | grep latency: | tail -n 94 | awk '{a+=$2}END{print a/NR}'
"""

if __name__ == "__main__":
    main()
