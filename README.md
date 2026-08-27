# WaveMLP: Cross-Receiver-Day RF Fingerprinting with Learnable Complex Wavelets

Official PyTorch implementation of **WaveMLP**, a lightweight architecture for **cross-receiver-day (CRD)** radio frequency fingerprinting (RFF).

> 📢 Accepted by **IMNS'26** (2026 International Conference on Intelligent Multimedia, Networking, and Security), to appear in the IEEE conference proceedings.

## Overview

Radio frequency fingerprinting identifies wireless devices through their hardware imperfections, but accuracy drops sharply once the receiver or the capture day differs between training and deployment — the **cross-receiver-day (CRD)** setting.

WaveMLP couples a **Learnable Complex Wavelet Transform (LCWT)** with a **lightweight MLP backbone**. The LCWT inherits the dyadic filter-bank structure of the wavelet transform for joint multi-resolution time–frequency analysis, while its complex analysis and synthesis filters are learned directly from data to suppress scenario distortions and preserve the device fingerprint.

On the **ManySig** and **ManyRx** subsets of WiSig, WaveMLP attains **93.13%** and **88.68%** average accuracy, surpassing seven representative baselines by **12.18%** and **2.67%**, with one to two orders of magnitude fewer parameters or FLOPs.

## Dataset

Experiments use the **WiSig** benchmark. Download it from the official source and place it under `./data/`:
https://cores.ee.ucla.edu/downloads/datasets/wisig/

| Subset  | Transmitters | Receivers | Samples / pair / day | Days | Sample length |
|---------|:-----------:|:---------:|:--------------------:|:----:|:-------------:|
| ManySig | 6           | 12        | 100                  | 4    | 2 × 256       |
| ManyRx  | 10          | 32        | 100                  | 4    | 2 × 256       |

**CRD protocol:** receivers are split into four disjoint groups for 4-fold cross-validation (3 train / 1 test); training uses the first two days, testing the last two.

## Usage

```bash
# Train (ManySig: J=4, K=16 | ManyRx: J=5, K=32)
python train.py --dataset manysig
python train.py --dataset manyrx

# Evaluate
python eval.py --dataset manysig --ckpt checkpoints/manysig_best.pth
```

Default setup: 200 epochs, Adam, learning rate `1e-3`, batch size `128`.

## Results

CRD recognition accuracy (%), average over 4 cross-validation rounds.

| Method   | ManySig | ManyRx | Params | FLOPs |
|----------|:-------:|:------:|:------:|:-----:|
| TFMix    | 80.95   | 73.47  | 2.59×10⁵ | 2.35×10⁷ |
| SigMix   | 80.01   | 73.90  | 2.59×10⁵ | 2.35×10⁷ |
| MTL      | 79.40   | 72.84  | 2.86×10⁴ | 2.57×10⁶ |
| FS       | 75.92   | 68.70  | 1.37×10⁶ | 1.88×10⁹ |
| RIEI     | 76.07   | 72.11  | 4.11×10⁶ | 8.81×10⁷ |
| GAN-RXA  | 78.45   | 74.33  | 2.02×10⁶ | 4.08×10⁷ |
| AWTD     | 71.05   | 86.01  | 1.12×10⁷ | 3.13×10⁷ |
| **WaveMLP** | **93.13** | **88.68** | **8.39/8.45×10⁴** | **1.27/1.40×10⁶** |

## Citation

```bibtex
@INPROCEEDINGS{11655284,
  author={Wang, Yu and Wang, Meiyu and Wang, Juzhen},
  booktitle={2026 International Conference on Intelligent Multimedia, Networking, and Security (IMNS)}, 
  title={WaveMLP: Cross-Receiver-Day RF Fingerprinting with Learnable Complex Wavelets}, 
  year={2026},
  volume={},
  number={},
  pages={1-5},
  keywords={Timing;Receivers;Fingerprint recognition;Filtering;Filters;Equations;Printing;Radio frequency;Transmitters;Learning (artificial intelligence);Radio frequency fingerprinting;crossreceiver-day;domain generalization;learnable complex wavelet transform},
  doi={10.1109/IMNS67862.2026.11655284}}
```

## Acknowledgment

This research was supported by the Joint Fund of Zhejiang Provincial Natural Science Foundation of China under Grant LLSQN25F010002 and the Zhejiang Province Postdoctoral Research Excellence Funding Project under Grant ZJ2025006.

## Contact

Yu Wang — `yuwang@njupt.edu.cn`
