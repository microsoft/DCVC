
# DCVC-family


:rocket: The DCVC-family offers a series of neural image and video codecs. This document provides links to the code, papers, and checkpoints for each codec. 

<table>
  <tr>
    <th>Model</th>
    <th>Paper</th>
    <th>Code</th>
    <th>Checkpoint</th>
  </tr>
  <tr>
    <td rowspan="2">DCVC</td>
    <td> 
      <a href="https://proceedings.neurips.cc/paper/2021/file/96b250a90d3cf0868c83f8c965142d2a-Paper.pdf">Paper (NeurIPS 2021)</a> & 
      <a href="https://arxiv.org/abs/2109.15047">Paper (arXiv)</a>
    </td>
    <td> <a href="DCVC">Code</a></td>
    <td><a href="https://1drv.ms/u/s!AozfVVwtWWYoiS5mcGX320bFXI0k?e=iMeykH">Checkpoints</a></td>
  </tr>
  <tr>
    <td colspan="3">
    Propose a paradigm shift from residual coding to conditional coding, i.e., <strong>D</strong>eep <strong>C</strong>ontextual <strong>V</strong>ideo <strong>C</strong>oding.
    </td>
  </tr>
  <tr>
    <td rowspan="2">DCVC-TCM</td>
    <td>
      <a href="https://ieeexplore.ieee.org/document/9941493">Paper (IEEE TMM)</a> & 
      <a href="https://arxiv.org/abs/2111.13850">Paper (arXiv)</a>
    </td>
    <td><a href="DCVC-TCM">Code</a></td>
    <td><a href="https://onedrive.live.com/?authkey=%21ADwwaonwTGR%5FNR8&id=2866592D5C55DF8C%211234&cid=2866592D5C55DF8C">Checkpoints</a></td>
  </tr>
  <tr>
    <td colspan="3">
    Propose a <strong>T</strong>emporal <strong>C</strong>ontext <strong>M</strong>ining method to extract more effective multi-scale temporal contexts for conditional coding.
    </td>
  </tr>
  <tr>
    <td rowspan="3">DCVC-HEM</td>
    <td>
      <a href="https://dl.acm.org/doi/abs/10.1145/3503161.3547845">Paper (ACM MM 2022)</a> & 
      <a href="https://arxiv.org/abs/2207.05894">Paper (arXiv)</a>
    </td>
    <td><a href="DCVC-HEM">Code</a></td>
    <td><a href="https://1drv.ms/u/s!AozfVVwtWWYoiUAGk6xr-oELbodn?e=kry2Nk">Checkpoints</a></td>
  </tr>
  <tr>
    <td colspan="3">
    Propose a <strong>H</strong>ybrid spatial-temporal <strong>E</strong>ntropy <strong>M</strong>odel to efficiently captures both spatial and temporal dependencies.
    </td>
  </tr>
  <tr>
    <td colspan="3">
    <li>The first end-to-end neural video codec to exceed H.266 (VTM) using the highest compression ratio low delay configuration with a intra refresh period of 32 frames, in terms of both PSNR and MS-SSIM.
    <li>The first end-to-end neural video codec to achieve rate adjustment in single model.
    </td>
  </tr>
  <tr>
    <td rowspan="3">DCVC-DC</td>
    <td>
      <a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Neural_Video_Compression_With_Diverse_Contexts_CVPR_2023_paper.pdf">Paper (CVPR 2023)</a> & 
      <a href="https://arxiv.org/abs/2302.14402">Paper (arXiv)</a>
    </td>
    <td><a href="DCVC-DC">Code</a></td>
    <td><a href="https://1drv.ms/u/s!AozfVVwtWWYoiWdwDhEkZMIfpon5?e=JcGri5">Checkpoints</a></td>
  </tr>
  <tr>
    <td colspan="3">
    Propose leveraging <strong>D</strong>iverse <strong>C</strong>ontexts in both temporal and spatial dimensions to enhance the performance of conditional coding.
    </td>
  </tr>
  <tr>
    <td colspan="3">
    <li> The first end-to-end neural video codec to exceed ECM using the highest compression ratio low delay configuration with a intra refresh period of 32 frames, in terms of PSNR for YUV420 content, and in terms of PSNR and MS-SSIM for RGB content.
    </td>
  </tr>
  <tr>
    <td rowspan="3">DCVC-FM</td>
    <td>
      <a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Neural_Video_Compression_with_Feature_Modulation_CVPR_2024_paper.pdf">Paper (CVPR 2024)</a> & 
      <a href="https://arxiv.org/abs/2402.17414">Paper (arXiv)</a>
    </td>
    <td><a href="DCVC-FM">Code</a></td>
    <td><a href="https://1drv.ms/f/s!AozfVVwtWWYoi1QkAhlIE-7aAaKV?e=OoemTr">Checkpoints</a></td>
  </tr>
  <tr>
    <td colspan="3">
    Propose a <strong>F</strong>eature <strong>M</strong>odulation scheme to support a wide quality range and operate over a long prediction chain.
    </td>
  </tr>
  <tr>
    <td colspan="3">
    <li> The first end-to-end neural video codec to exceed ECM using the highest compression ratio low delay configuration with only one intra frame, in terms of PSNR for both YUV420 content and RGB content in a single model. 
    <li> The first end-to-end neural video codec that support a large quality and bitrate range in a single model.
    </td>
  </tr>
  <tr>
    <td rowspan="3">DCVC-RT</td>
    <td>
      <a href="https://arxiv.org/abs/2502.20762">Paper (arXiv)</a>
    </td>
    <td><a href="https://github.com/microsoft/DCVC/tree/main">Code</a></td>
    <td><a href="https://1drv.ms/f/c/2866592d5c55df8c/Esu0KJ-I2kxCjEP565ARx_YB88i0UnR6XnODqFcvZs4LcA?e=by8CO8">Checkpoints</a></td>
  </tr>
  <tr>
    <td colspan="3">
    Propose a <strong>R</strong>eal-<strong>T</strong>ime neural video codec with efficiency-driven design improvements to minimize operational costs.
    </td>
  </tr>
  <tr>
    <td colspan="3">
    <li> The first end-to-end neural video codec achieving 100+ FPS 1080p coding and 4K real-time coding with a comparable compression ratio with ECM.
    </td>
  </tr>
  <tr>
    <td rowspan="3">EVC</td>
    <td>
      <a href="https://openreview.net/forum?id=XUxad2Gj40n">Paper (ICLR 2023)</a> & 
      <a href="https://arxiv.org/abs/2302.05071">Paper (arXiv)</a>
    </td>
    <td><a href="EVC">Code</a></td>
    <td><a href="https://1drv.ms/u/s!AozfVVwtWWYoiUhZLZDx7vJjHK1C?e=qETpA1">Checkpoints</a></td>
  </tr>
  <tr>
    <td colspan="3">
    Propose a mask decay method to transform a neural image codec into a small <strong>E</strong>ffective <strong>V</strong>ariable-bit-rate <strong>Codec</strong>.
    </td>
  </tr>
  <tr>
    <td colspan="3">
    <li> Achieves 30 FPS real-time coding for 768x512 images while surpassing VVC intra coding performance.
    </td>
  </tr>
</table>


* As a backup, all the pretrained models could be found [here](https://1drv.ms/f/c/2866592d5c55df8c/EozfVVwtWWYggCitBAAAAAABbT4z2Z10fMXISnan72UtSA?e=BID7DA).

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

@inproceedings{jia2025towards,
  title={Towards Practical Real-Time Neural Video Compression},
  author={Jia, Zhaoyang and Li, Bin and Li, Jiahao and Xie, Wenxuan and Qi, Linfeng and Li, Houqiang and Lu, Yan},
  booktitle={{IEEE/CVF} Conference on Computer Vision and Pattern Recognition,
             {CVPR} 2025, Nashville, TN, USA, June 11-25, 2024},
  year={2025}
}

@inproceedings{wang2023EVC,
  title={EVC: Towards Real-Time Neural Image Compression with Mask Decay},
  author={Wang, Guo-Hua and Li, Jiahao and Li, Bin and Lu, Yan},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```