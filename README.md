# mmWave Wireless Sensing — 3D Human Pose Estimation

An end-to-end deep learning pipeline for non-intrusive 3D human pose estimation using millimetre-wave (mmWave) radar — the same sensing modality used in modern smartphones, wearables, and autonomous vehicles. Unlike camera-based approaches, mmWave preserves privacy and works in low-light, occluded, and through-wall environments.

Based on the [MMFi dataset and benchmark framework](https://arxiv.org/pdf/2308.03149) (Yang et al., 2023).

---

## Problem

mmWave radar produces extremely sparse 3D point clouds (~100 points per frame vs. millions in RGB imagery). This sparsity makes it hard for ML models to extract meaningful spatial features for pose estimation — standard convolution architectures underperform without careful temporal aggregation.

---

## Approach

**Dataset:** Mini-MMFI (MARS) — 10 subjects, 27 activities including squats, jumps, lateral lunges, and full-body extensions.

**MFPC (Multi-Frame Point Cloud):** Designed a voxelisation pipeline that fuses 5 consecutive radar frames into a dense 3D spatial grid, directly addressing sparsity by accumulating signal across time before feature extraction.

**Preprocessing pipeline:**
- Intensity-based filtering to remove low-confidence returns
- Spatial centering and normalisation per sequence
- Frame-level quality filtering to drop corrupted scans

**Models built and compared:**

| Architecture | Design |
|---|---|
| Temporal CNN | TimeDistributed conv layers + GlobalAveragePooling over the MFPC sequence |
| CNN-LSTM | Same CNN backbone with an LSTM head for explicit temporal modelling |

**Denoising ablation:** Evaluated moving average, Gaussian smoothing, and threshold filtering applied on top of MFPC features to assess signal quality impact on joint prediction.

---

## Results

**Task:** Predict x, y, z coordinates of 17 body joints simultaneously (51 output values per frame).

| Model | MAE ↓ | RMSE ↓ | Val Loss (epoch 30) |
|---|---|---|---|
| CNN (no denoising) | 0.1293 | 0.1795 | 0.0412 |
| CNN + moving average | **0.1241** | **0.1718** | **0.0389** |
| CNN + Gaussian filter | 0.1268 | 0.1751 | 0.0401 |
| CNN-LSTM | 0.3165 | 0.4207 | 0.1823 |

The temporal CNN with moving average denoising outperformed all variants. The CNN-LSTM showed significant overfitting after ~5 epochs (training MAE 0.009 vs. validation MAE 0.40 at epoch 30), pointing to dataset scale as the primary bottleneck for recurrent architectures.

---

## Stack

Python · TensorFlow/Keras · NumPy · Matplotlib · scikit-learn

---

## What I'd improve next

- **Attention over time:** Replace the LSTM with a transformer-style temporal attention module — better long-range dependencies, less prone to overfitting on small datasets
- **PointNet backbone:** Process raw point clouds directly rather than voxelising, preserving spatial resolution lost in the grid projection
- **Transfer learning:** Pre-train on full MMFi (40 subjects) before fine-tuning on the mini split
- **Real-time inference:** Quantise and optimise for embedded deployment on edge devices (Raspberry Pi, NVIDIA Jetson)

---

## Applications

Healthcare & fall detection · Automotive in-cabin gesture control · Smart home presence sensing · Sports biomechanics · Through-wall surveillance
