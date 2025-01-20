# Introduction

Official Pytorch implementation for Neural Video and Image Compression including:
* Neural Video Codec
  * DCVC: [Deep Contextual Video Compression](https://proceedings.neurips.cc/paper/2021/file/96b250a90d3cf0868c83f8c965142d2a-Paper.pdf), NeurIPS 2021, in [this folder](./DCVC/).
  * DCVC-TCM: [**T**emporal **C**ontext **M**ining for Learned Video Compression](https://ieeexplore.ieee.org/document/9941493), in IEEE Transactions on Multimedia, and [arxiv](https://arxiv.org/abs/2111.13850), in [this folder](./DCVC-TCM/).
  * DCVC-HEM: [**H**ybrid Spatial-Temporal **E**ntropy **M**odelling for Neural Video Compression](https://arxiv.org/abs/2207.05894), ACM MM 2022, in [this folder](./DCVC-HEM/).
    -  The first end-to-end neural video codec to exceed H.266 (VTM) using the highest compression ratio configuration, in terms of both PSNR and MS-SSIM.
    -  The first end-to-end neural video codec to achieve rate adjustment in single model.
  * DCVC-DC: [Neural Video Compression with **D**iverse **C**ontexts](https://arxiv.org/abs/2302.14402), CVPR 2023, in [this folder](./DCVC-DC/).
    -  The first end-to-end neural video codec to exceed [ECM](https://jvet-experts.org/doc_end_user/documents/27_Teleconference/wg11/JVET-AA0006-v1.zip) using the highest compression ratio low delay configuration with a intra refresh period roughly to one second (32 frames), in terms of PSNR and MS-SSIM for RGB content.
    -  The first end-to-end neural video codec to exceed ECM using the highest compression ratio low delay configuration with a intra refresh period roughly to one second (32 frames), in terms of PSNR for YUV420 content.
  * DCVC-FM: [Neural Video Compression with **F**eature **M**odulation](https://arxiv.org/abs/2402.17414), CVPR 2024, in [this folder](./DCVC-FM/).
    -  The first end-to-end neural video codec to exceed ECM using the highest compression ratio low delay configuration with only one intra frame, in terms of PSNR for both YUV420 content and RGB content in a single model.
    -  The first end-to-end neural video codec that support a large quality and bitrate range in a single model.
* Neural Image Codec
  * [EVC: Towards Real-Time Neural Image Compression with Mask Decay](https://openreview.net/forum?id=XUxad2Gj40n), ICLR 2023, in [this folder](./EVC/).

# Pretrained models

As a backup, all the pretrained models could be found [here](https://1drv.ms/f/c/2866592d5c55df8c/EozfVVwtWWYggCitBAAAAAABbT4z2Z10fMXISnan72UtSA?e=BID7DA).

# On the comparison

Please note that different methods may use different configurations to test different models, such as
* Source video may be different, e.g., cropped or padded to the desired resolution.
* Intra period may be different, e.g., 96, 32, 12, or 10.
* Number of encoded frames may be different.

So, it does not make sense to compare the numbers in different methods directly, unless making sure they are using same test conditions.

Please find more details on the [test conditions](./test_conditions.md).

# Acknowledgement
The implementation is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI) and [PyTorchVideoCompression](https://github.com/ZhihaoHu/PyTorchVideoCompression).

# Citation
If you find this work useful for your research, please cite:

```
@article{li2021deep,
  title={Deep Contextual Video Compression},
  author={Li, Jiahao and Li, Bin and Lu, Yan},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}

@article{sheng2022temporal,
  title={Temporal context mining for learned video compression},
  author={Sheng, Xihua and Li, Jiahao and Li, Bin and Li, Li and Liu, Dong and Lu, Yan},
  journal={IEEE Transactions on Multimedia},
  year={2022},
  publisher={IEEE}
}

@inproceedings{li2022hybrid,
  title={Hybrid Spatial-Temporal Entropy Modelling for Neural Video Compression},
  author={Li, Jiahao and Li, Bin and Lu, Yan},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  year={2022}
}

@inproceedings{li2023neural,
  title={Neural Video Compression with Diverse Contexts},
  author={Li, Jiahao and Li, Bin and Lu, Yan},
  booktitle={{IEEE/CVF} Conference on Computer Vision and Pattern Recognition,
             {CVPR} 2023, Vancouver, Canada, June 18-22, 2023},
  year={2023}
}

@inproceedings{li2024neural,
  title={Neural Video Compression with Feature Modulation},
  author={Li, Jiahao and Li, Bin and Lu, Yan},
  booktitle={{IEEE/CVF} Conference on Computer Vision and Pattern Recognition,
             {CVPR} 2024, Seattle, WA, USA, June 17-21, 2024},
  year={2024}
}

@inproceedings{wang2023EVC,
  title={EVC: Towards Real-Time Neural Image Compression with Mask Decay},
  author={Wang, Guo-Hua and Li, Jiahao and Li, Bin and Lu, Yan},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

# Trademarks
This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.
