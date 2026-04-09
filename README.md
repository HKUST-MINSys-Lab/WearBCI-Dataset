# WearBCI Dataset: Understanding and Benchmarking Real-World Wearable Brain-Computer Interfaces Signals

The multimodal wearable BCI dataset with synchronized EEG, IMU, and egocentric video, collected from 36 participants across motion dynamics of increasing complexity.

This repository provides the code release of [**WearBCI**](https://doi.org/10.1145/3774906.3802782). WearBCI includes synchronized 8-channel EEG (250 Hz), 5-point IMU (100 Hz), and egocentric video (30 FPS) recordings across four sessions: static baseline, body movements, walking, and navigation. Beyond benchmarking existing EEG enhancement methods, the dataset enables two case studies: cross-modal EEG signal enhancement using IMU, and multi-dimensional human behavior understanding using multimodal large language models.

✅ Paper: [ACM SenSys '26](https://doi.org/10.1145/3774906.3802782) \
✅ Dataset: [WearBCI Dataset](https://1drv.ms/f/c/6a75cf94601a650a/IgAaS_6cbyENT6tmgPaIXDLtARspJ7FwXkK14fj88JZ_zEY?e=beEqvh)

## Dataset Overview

| | |
|:---|:---|
| **Subjects** | 36 healthy adults (ages 18–26), balanced gender ratio |
| **Total Duration** | EEG: 16.4 h, IMU: 9.2 h, Video: 0.9 h |
| **Modalities** | 8-ch EEG (250 Hz), 5 IMUs (100 Hz), egocentric video (30 FPS) |
| **Sessions** | Static, Body Movements (5 types), Walking (3 speeds), Navigation |

**IMU placements**: head, left wrist, right wrist, left ankle, right ankle. \
**EEG electrodes**: Fp1, Fp2, P7, P8, T3, T4, O1, O2 (10-20 system, OpenBCI Cyton board).

## Repository Structure

```text
├── eeg_signal_enhancement/     # Case Study 1: Cross-Modal EEG Signal Enhancement
└── behavior_understanding/     # Case Study 2: Multi-dimension Behavior Understanding
```

---

## Case Study 1: Cross-Modal EEG Signal Enhancement

IMU signals from five body-worn sensors are used to estimate and subtract motion-induced artifacts from raw EEG. A linear model minimizes a combined temporal MSE and frequency-domain PSD loss, encouraging both waveform fidelity and spectral alignment with a static baseline.

### Quick Start

#### 1. Installation
```bash
cd eeg_signal_enhancement
pip install -r requirements.txt
```

#### 2. Dataset preparation
Download the WearBCI dataset and place the walking session data under `eeg_signal_enhancement/data/`:
- EEG walking recordings → `data/eeg/`
- IMU recordings per placement → `data/imu/<placement>/` (head / left_wrist / right_wrist / left_ankle / right_ankle)

#### 3. Run
```bash
python main.py
```

Results are written to `results/`:
- `<task>_timeseries.png` — time-series comparison (Noisy / Cleaned / Baseline)
- `<task>_denoised.csv` — denoised EEG signal

---

## Case Study 2: Multi-dimension Behavior Understanding

Synchronized EEG, IMU, and egocentric video from the navigation session are analyzed by Gemini 2.5 Pro to infer three semantic dimensions of behavior: **Scene** (environmental context), **Action** (motor behavior), and **Cognition** (mental engagement). Each modality is analyzed independently, then combined via two fusion strategies — Concatenate Fusion (early/pre-fusion) and Optimized Fusion (late/post-fusion with cross-modal conflict reconciliation) — and evaluated against human annotations using BERTScore F1.

### Quick Start

#### 1. Installation
```bash
cd behavior_understanding
pip install pandas numpy matplotlib scipy opencv-python requests bert-score torch
```

#### 2. Dataset preparation
Download the WearBCI dataset and place the navigation session data under `behavior_understanding/data/`:
- EEG navigation recording → `data/eeg/`
- IMU recordings per placement → `data/imu/<placement>/` (head / left_wrist / right_wrist / left_ankle / right_ankle)
- Egocentric video → `data/video/`

#### 3. Configuration
```bash
cp config/.env.example config/.env
# Edit config/.env and fill in your Gemini API key
```

#### 4. Run

Run a single modality:
```bash
python main.py --api_key YOUR_GEMINI_API_KEY --mode eeg
python main.py --api_key YOUR_GEMINI_API_KEY --mode imu
python main.py --api_key YOUR_GEMINI_API_KEY --mode video
```

Run multimodal fusion:
```bash
python main.py --api_key YOUR_GEMINI_API_KEY --mode concatenate
python main.py --api_key YOUR_GEMINI_API_KEY --mode optimized
```

Run evaluation against ground truth annotations:
```bash
python main.py --api_key YOUR_GEMINI_API_KEY --mode eval
```

Run the full pipeline:
```bash
python main.py --api_key YOUR_GEMINI_API_KEY
```

---

## Citation

Please consider citing our paper if you use this dataset or code in your research.

```bibtex
@inproceedings{liu2026wearbci,
  title     = {WearBCI Dataset: Understanding and Benchmarking Real-World Wearable Brain-Computer Interfaces Signals},
  author    = {Liu, Haoxian and Jiang, Hengle and Hong, Lanxuan and Ouyang, Xiaomin},
  booktitle = {Proceedings of the 24th ACM/IEEE International Conference on Embedded Artificial Intelligence and Sensing Systems (SenSys '26)},
  year      = {2026},
  address   = {Saint-Malo, France},
  publisher = {ACM},
  doi       = {10.1145/xxxxxxx.xxxxxxx}
}
```
