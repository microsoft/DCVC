# Suggested test conditions for neural video codec

We find that different papers on neural video codec use quite different conditions. Some of them may use test conditions far away from the real applications, such as using a very small intra period of 10 or 12. Some papers may even compare different schemes directly without confirming the test conditions are identical. To make the comparison fair and the test conditions close to practical applications, we suggest the following test conditions for neural video codec.

## Test sequences

HEVC test sequences, [UVG](http://ultravideo.fi/#testsequences), and [MCL-JCV](http://mcl.usc.edu/mcl-jcv-dataset/) are widely used in both traditional codecs and neural codecs. Most of the source sequences are in YUV420 format. Using YUV420 as the source content is ideal and we suggest encoding YUV420 sequences when possible.

However, most of neural codecs uses RGB as the input. Thus, we suggest using the BT.709 full color range to convert YUV420 sequences to RGB (png). More details could be found from the suggested test conditions of [JPEG-AI](https://jpegai.github.io/7-anchors/).

## Encoding settings

There are several key factors in encoding setting:

* Please do not crop source sequence. When the spatial resolution of the source sequence is not supported, consider padding it.
* Number of encoded frames should not be too small. Considering the test sequence length, we suggest encode 96 frames in each sequence. (The shortest sequence in MCL-JCV dataset has 120 frames).
* Set to a reasonable intra period. Some of the previous papers uses 10 or 12 as intra period, which is not reasonable in practical applications. We suggest using 32 as intra period. When possible, setting it to 96 (only the first frame is encoded as intra frame) is also desired.
* Make sure the intra frame is pure intra frame. Which can be decoded independent of other frames. Intra frame could also be used as a decoding refresh point.
* Both low delay coding structure (without waiting for future frames when encoding/decoding current frame) and hierarchical-B coding structure are welcome to be tested.

Here is the summary on the coding setting

|                    | Not suggested | Suggested |
| ------------------ | ------------- | --------- |
| spatial resolution | crop          | pad       |
| intra period       | 10, 12        | 32, 96    |

## Traditional codec anchors

When comparing with traditional codecs, it is highly suggested to use the settings which could lead to the highest compression ratio for traditional codec. There are several key factors for traditional codecs:

* Use B frames (or low delay B frames) instead of P frames.
* Use the suggested number of reference frames (e.g., 4 for low delay settings).
* Use hierarchical QP settings. (all the three items above could lead to 27% gap)
* Use 10-bit as internal bit-depth. (6% gap over 8-bit)
* Use YUV420 when comparing for YUV420 source content and use YUV444 when comparing RGB content. (22% gap when using different color spaces)

We suggest using the official reference software (HM, VTM, and ECM) to generate tradition codec anchors. For low delay encoding, the suggested command lines are as follows. We suggest using HM-16.25, VTM-17.0, and ECM-5.0 as traditional codec reference. Using a more recent version is also preferred.

### Encoding for RGB content

YUV444 as input when comparing RGB content. We suggest convert RGB (png) to YUV444 10-bit first. The conversion details could be found from the suggested test conditions of [JPEG-AI](https://jpegai.github.io/7-anchors/).

Please note that ECM-5.0 has several bugs to support YUV444 encoding. You may need to fix the bugs before using it to encode YUV444 content.

| encoder  | configuration file             |
| -------- | ------------------------------ |
| HM-16.25 | encoder_lowdelay_main_rext.cfg |
| VTM-17.0 | encoder_lowdelay_vtm.cfg       |
| ECM-5.0  | encoder_lowdelay_ecm.cfg       |

```bash
EncoderApp -c encoder_configuration.cfg -f 96 -q {qp} --IntraPeriod={intra_period} --InputFile={src_yuv} --SourceWidth={width} --SourceHeight={height} --FrameRate={frame_rate} --Level=6.2 --InputBitDepth=10 --InputChromaFormat=444 --ChromaFormatIDC=444 --DecodingRefreshType=2 -b {output.bin} -o {enc.yuv}
```

### Encoding for YUV420 content

| encoder  | configuration file          |
| -------- | --------------------------- |
| HM-16.25 | encoder_lowdelay_main10.cfg |
| VTM-17.0 | encoder_lowdelay_vtm.cfg    |
| ECM-5.0  | encoder_lowdelay_ecm.cfg    |

```bash
EncoderApp -c encoder_configuration.cfg -f 96 -q {qp} --IntraPeriod={intra_period} --InputFile={src_yuv} --SourceWidth={width} --SourceHeight={height} --FrameRate={frame_rate} --Level=6.2 --InputBitDepth=8 --DecodingRefreshType=2 -b {output.bin} -o {enc.yuv}
```

## Objective metrics

We suggest using PSNR and MS-SSIM as objective metrics when comparing different methods. For RGB content the PSNR and MS-SSIM should be calculated in RGB domain. For YUV420 content, PSNR and MS-SSIM should be calculated for Y, U, and V separately. It is also suggested using the following equation to get the average PSNR or MS-SSIM for YUV420 content.

```math
PSNR_{avg} = (6*PSNR_y + PSNR_u + PSNR_v) / 8
```
