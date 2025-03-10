# Introduction
* Official Pytorch implementation for DCVC-TCM: [**T**emporal **C**ontext **M**ining for Learned Video Compression](https://ieeexplore.ieee.org/document/9941493), in IEEE Transactions on Multimedia.
* Arxiv version can be found [here](https://arxiv.org/abs/2111.13850).

# Prerequisites
* Python 3.6 and conda, get [Conda](https://www.anaconda.com/)
* CUDA if want to use GPU
* pytorch==1.10.0
* torchvision==0.11.0
* cudatoolkit=11.1
* Other tools
    ```
    pip install -r requirements.txt
    ```
# Test dataset
We support arbitrary original resolution. The input video resolution will be padded to 64x automatically. The reconstructed video will be cropped back to the original size. The distortion (PSNR/MS-SSIM) is calculated at original resolution.

The dataset format can be seen in dataset_config_example.json.

For example, one video of HEVC Class B can be prepared as:
* Make the video path:
    ```
    mkdir BasketballDrive_1920x1080_50
    ```
* Convert YUV to PNG:
    ```
    ffmpeg -pix_fmt yuv420p -s 1920x1080 -i BasketballDrive_1920x1080_50.yuv -f image2 BasketballDrive_1920x1080_50/im%05d.png
    ```
At last, the folder structure of dataset is like:

    /media/data/HEVC_B/
        * BQTerrace_1920x1080_60/
            - im00001.png
            - im00002.png
            - im00003.png
            - ...
        * BasketballDrive_1920x1080_50/
            - im00001.png
            - im00002.png
            - im00003.png
            - ...
        * ...
    /media/data/HEVC_D
    /media/data/HEVC_C/
    ...

# Build the project
Please build the C++ code if want to test with actual bitstream writing. There is minor difference about the bits for calculating the bits using entropy (the method used in the paper to report numbers) and actual bitstreaming writing. There is overhead when writing the bitstream into the file and the difference percentage depends on the bitstream size. Usually, the overhead for 1080p content is less than 0.5%.
## On Windows
```bash
cd src
mkdir build
cd build
conda activate $YOUR_PY38_ENV_NAME
cmake ../cpp -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

## On Linux
```bash
sudo apt-get install cmake g++
cd src
mkdir build
cd build
conda activate $YOUR_PY36_ENV_NAME
cmake ../cpp -DCMAKE_BUILD_TYPE=Release
make -j
```

# Pretrained Models

* Download our [pretrained models](https://onedrive.live.com/?authkey=%21ADwwaonwTGR%5FNR8&id=2866592D5C55DF8C%211234&cid=2866592D5C55DF8C) and put them into ./checkpoints folder.

# Test
Use ```test_video.py``` to test the models and the results will be saved to a JSON file. You need to specify the paths to the trained I-frame models and P-frame models, as well as a test configuration file, for example, ```recommended_test_full_results_IP12.json```. A complete example is given below:
```bash
python test_video.py --i_frame_model_name IntraNoAR --i_frame_model_path ./checkpoints/intra_psnr_TMM_q1.pth.tar --model_path ./checkpoints/inter_psnr_TMM_q1.pth --test_config recommended_test_full_results_IP12.json --cuda 1 -w 1 --write_stream 0 --output_path results_IP12.json
```

# Acknowledgement
The implementation is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI) and [PyTorchVideoCompression](https://github.com/ZhihaoHu/PyTorchVideoCompression).
# Citation
If you find this work useful for your research, please cite:

```
@article{sheng2022temporal,
  title={Temporal context mining for learned video compression},
  author={Sheng, Xihua and Li, Jiahao and Li, Bin and Li, Li and Liu, Dong and Lu, Yan},
  journal={IEEE Transactions on Multimedia},
  year={2022},
  publisher={IEEE}
}
```
