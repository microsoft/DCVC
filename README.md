# Introduction

Official Pytorch implementation for Neural Video Compression including:
* [Deep Contextual Video Compression](https://proceedings.neurips.cc/paper/2021/file/96b250a90d3cf0868c83f8c965142d2a-Paper.pdf), NeurIPS 2021, in [this folder](./NeurIPS2021/).
* [Hybrid Spatial-Temporal Entropy Modelling for Neural Video Compression](https://arxiv.org/abs/2207.05894), ACM MM 2022, in [this folder](./ACMMM2022/).
  -  The first end-to-end neural video codec to exceed H.266 (VTM) using the highest compression ratio configuration, in terms of both PSNR and MS-SSIM.
  -  The first end-to-end neural video codec to achieve rate adjustment in single model.

# On the comparison

Please note that different methods may use different configurations to test different models, such as
* Source video may be different, e.g., cropped or padded to the desired resolution.
* Intra period may be different, e.g., 32, 12, or 10.
* Number of encoded frames may be different.

So, it does not make sense to compare the numbers in different methods directly, unless making sure they are using same test conditions.

Please find more details on the [test_conditions](./test_conditions.md).

# Command line to generate VTM results

Get VTM from https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM and build the project.
```bash
EncoderApp -c encoder_lowdelay_vtm.cfg --InputFile={input file name} --BitstreamFile={bitstream file name} --DecodingRefreshType=2 --InputBitDepth=8 --OutputBitDepth=8 --OutputBitDepthC=8 --InputChromaFormat=444 --FrameRate={frame rate} --FramesToBeEncoded={frame number} --SourceWidth={width} --SourceHeight={height} --IntraPeriod=32 --QP={qp} --Level=6.2
```

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

@inproceedings{li2022hybrid,
  title={Hybrid Spatial-Temporal Entropy Modelling for Neural Video Compression},
  author={Li, Jiahao and Li, Bin and Lu, Yan},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  year={2022}
}
```

# Trademarks
This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.
