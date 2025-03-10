
# DCVC-family


:rocket: The DCVC-family offers a series of neural image and video codecs. This document provides links to the code, papers, and checkpoints for each codec. 

| Model | Link | Checkpoint |
| ----- | ---- | ---------- |
| DCVC  | [Code](DCVC) & [Paper (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/file/96b250a90d3cf0868c83f8c965142d2a-Paper.pdf) & [Paper (arXiv)](https://arxiv.org/abs/2109.15047)| [Checkpoints](https://1drv.ms/u/s!AozfVVwtWWYoiS5mcGX320bFXI0k?e=iMeykH) |
| DCVC-TCM  | [Code](DCVC-TCM) & [Paper (IEEE TMM)](https://ieeexplore.ieee.org/document/9941493) & [Paper (arXiv)](https://arxiv.org/abs/2111.13850)| [Checkpoints](https://onedrive.live.com/?authkey=%21ADwwaonwTGR%5FNR8&id=2866592D5C55DF8C%211234&cid=2866592D5C55DF8C) |
| DCVC-HEM  | [Code](DCVC-HEM) & [Paper (ACM MM 2022)](https://dl.acm.org/doi/abs/10.1145/3503161.3547845) & [Paper (arXiv)](https://arxiv.org/abs/2207.05894)| [Checkpoints](https://1drv.ms/u/s!AozfVVwtWWYoiUAGk6xr-oELbodn?e=kry2Nk) |
| DCVC-DC  | [Code](DCVC-DC) & [Paper (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Neural_Video_Compression_With_Diverse_Contexts_CVPR_2023_paper.pdf) & [Paper (arXiv)](https://arxiv.org/abs/2302.14402)| [Checkpoints](https://1drv.ms/u/s!AozfVVwtWWYoiWdwDhEkZMIfpon5?e=JcGri5) |
| DCVC-FM  | [Code](DCVC-FM) & [Paper (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Neural_Video_Compression_with_Feature_Modulation_CVPR_2024_paper.pdf) & [Paper (arXiv)](https://arxiv.org/abs/2402.17414)| [Checkpoints](https://1drv.ms/f/s!AozfVVwtWWYoi1QkAhlIE-7aAaKV?e=OoemTr) |
| EVC  | [Code](EVC) & [Paper (ICLR 2023)](https://openreview.net/forum?id=XUxad2Gj40n) & [Paper (arXiv)](https://arxiv.org/abs/2302.05071)| [Checkpoints](https://1drv.ms/u/s!AozfVVwtWWYoiUhZLZDx7vJjHK1C?e=qETpA1) |

* As a backup, all the pretrained models could be found [here](https://1drv.ms/f/c/2866592d5c55df8c/EozfVVwtWWYggCitBAAAAAABbT4z2Z10fMXISnan72UtSA?e=BID7DA).

:bouquet: The implementation of DCVC-family is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI) and [PyTorchVideoCompression](https://github.com/ZhihaoHu/PyTorchVideoCompression).

:page_facing_up: If you find this work useful for your research, please cite:

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