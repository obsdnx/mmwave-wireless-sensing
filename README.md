# mmWave Wireless Sensing — 3D Human Pose Estimation

An end-to-end deep learning pipeline for non-intrusive 3D human pose estimation using millimetre-wave (mmWave) radar. Unlike camera-based approaches, mmWave preserves privacy, works in low-light and occluded environments, and operates through non-metallic barriers.

Based on the [MM-Fi dataset and benchmark framework](https://arxiv.org/pdf/2308.03149) (Yang et al., NeurIPS 2023) and the MARS assistive rehabilitation system (An & Ogras, ACM TECS 2021).

---

## Background: mmWave Sensing

Millimetre-wave radar operates in the 30–300 GHz frequency band. This high-frequency operation provides:

- **Extremely high spatial resolution** — detects minute movements and enables precise object localisation
- **Compact form factor** — short wavelength allows small antennas suitable for embedded devices
- **Privacy preservation** — no optical imagery, no identifiable features captured

Most mmWave systems use **Frequency-Modulated Continuous Wave (FMCW)** radar, which transmits a continuously frequency-swept signal called a chirp. The system estimates three quantities per reflection:

| Quantity | Mechanism |
|---|---|
| Range | Frequency difference between transmitted and received chirp (Intermediate Frequency) |
| Velocity | Phase difference between successive chirps |
| Angle | Phase difference across multiple receive antennas (MIMO array) |

Compared to Wi-Fi sensing (2.4/5 GHz, uses Channel State Information), mmWave offers significantly higher resolution and 3D sensing capability at the cost of shorter range and higher atmospheric attenuation.

---

## Problem

mmWave radar produces extremely sparse 3D point clouds — on average ~100 points per frame versus millions in RGB imagery. This sparsity makes it hard for ML models to extract meaningful spatial features. Standard convolution architectures underperform without careful temporal aggregation.

---

## Dataset: mini-MMFI (MARS)

An abridged version of the dataset from the [MARS paper](https://doi.org/10.1145/3477003) (An & Ogras, ACM 2021).

| Property | Detail |
|---|---|
| Participants | 10 subjects (S01–S10) |
| Activities | 27 distinct actions (A01–A27), including limb extensions, squats, and jumps |
| Modalities | Simultaneous mmWave radar + RGB camera |
| mmWave usage | Input to CNN pipeline |
| RGB usage | Ground truth skeleton generation |

**Directory structure:**

```
DB_Coursework/
├── S01/
│   ├── A01/
│   │   ├── mmWave/          # raw .bin point cloud frames
│   │   └── ground_truth.npy # 3D joint positions (17 joints × 3 coords)
│   ├── A02/ ...
│   └── A27/
├── S02/ ...
└── S10/
```

Each ground truth file contains 17 joint positions with x, y, z coordinates (51 values per frame).

---

## Methodology

### MFPC Feature Representation

**Multi-Frame Point Cloud (MFPC)** addresses data sparsity by fusing multiple consecutive radar frames into a single dense 3D voxel grid. Proposed originally by An et al. in [4].

A single mmWave frame contains ~100 points; MFPC aggregates 5 frames into a sliding window, producing a richer spatial representation that also captures temporal motion context:

```
F[k] = {f[k-1], f[k], f[k+1], ...}  (window of 5 frames)
```

Benefits:
- Reveals finer detail obscured in single sparse frames
- Mitigates noise and frame-to-frame inconsistencies
- Inherently encodes temporal context
- Compatible with standard 3D object detection pipelines

**Core implementation:**

```python
def create_mfpc_features(mmwave_data, grid_size=32, grid_limit=10, frame_window=5):
    scale = grid_size / (2 * grid_limit)
    mfpc_features = []

    for i in range(len(mmwave_data) - frame_window + 1):
        aggregated_frames = []
        for j in range(frame_window):
            grid = np.zeros((grid_size, grid_size, grid_size))
            for point in mmwave_data[i + j]:
                x, y, z = point[:3] * scale + grid_size // 2
                if 0 <= x < grid_size and 0 <= y < grid_size and 0 <= z < grid_size:
                    grid[int(x), int(y), int(z)] += 1
            aggregated_frames.append(grid)
        mfpc_features.append(np.array(aggregated_frames))

    return np.array(mfpc_features)
```

Each point is mapped to a 32×32×32 voxel grid. The voxel count accumulates signal density across the 5-frame window before passing to the model.

### Preprocessing Pipeline

Applied before MFPC construction:

| Step | Purpose |
|---|---|
| Normalisation | Uniform scaling of point cloud values across frames |
| Intensity filtering | Removes low-intensity returns (noise / irrelevant reflections) |
| Centering | Shifts each frame so its mean position is at the origin |
| Frame filtering | Drops frames with invalid or incomplete data |

### Denoising Ablation

Three denoising techniques were evaluated on top of MFPC features:

- **Moving average** — arithmetic mean over a sliding window; attenuates stochastic noise while preserving the fundamental signal
- **Gaussian smoothing** — convolves with a Gaussian kernel; suppresses high-frequency noise with distance-weighted averaging
- **Threshold denoising** — nullifies values below a set amplitude threshold; eliminates low-amplitude noise under the assumption that signal >> noise

---

## Model Architectures

### Model 1: Temporal CNN (`train_cnn.py`)

| Layer | Configuration |
|---|---|
| Input reshaping | Reshape to (frames, H, W, C) |
| Feature extraction | `TimeDistributed(Conv2D(16, 3×3, ReLU, same))` |
| Regularisation | `TimeDistributed(Dropout(0.3))` |
| Flattening | `TimeDistributed(Flatten())` |
| Dense ×3 | `Dense(512, ReLU)` → `BatchNormalization()` → `Dropout(0.4)` |
| Temporal aggregation | `GlobalAveragePooling1D()` |
| Output | `Dense(51, linear)` — 17 joints × (x, y, z) |

The `TimeDistributed` wrapper applies the same 2D convolution independently to each frame in the window, preserving temporal structure. `GlobalAveragePooling1D` then collapses the temporal dimension into a single fixed-size vector, providing translation invariance over time.

**Training config:** Adam (lr=0.001) · MSE loss · batch size 16 · 30 epochs · 80/20 train/val split

### Model 2: CNN-LSTM (`train_cnn_lstm.py`)

Same CNN backbone as Model 1, with LSTM layers replacing the GlobalAveragePooling for explicit temporal modelling:

| Layer | Configuration |
|---|---|
| Feature extraction | `TimeDistributed(Conv2D(16, 3×3, ReLU, same))` |
| Regularisation | `TimeDistributed(Dropout(0.3))` |
| Flattening | `TimeDistributed(Flatten())` |
| LSTM layer 1 | `LSTM(128, return_sequences=True)` |
| LSTM layer 2 | `LSTM(64, return_sequences=False)` |
| Dense | `Dense(512, ReLU)` → `BatchNormalization()` → `Dropout(0.4)` |
| Output | `Dense(51, linear)` |

LSTM gating mechanisms selectively retain or forget information over long sequences, addressing the vanishing gradient problem of standard RNNs. The two-layer stacking allows the model to learn hierarchical temporal representations.

**Training config:** Identical to CNN — Adam (lr=0.001) · MSE loss · batch size 16 · 30 epochs

---

## Results

**Evaluation metrics:** MSE (training objective), MAE (interpretable error magnitude), RMSE (comparable to literature values)

**Data split:** Training — 16 files · Testing — 10 files

### CNN — Denoising Method Comparison

| Denoising Method | Loss (MSE) | MAE | RMSE |
|---|---|---|---|
| None (baseline) | 0.0322 | 0.1293 | 0.1795 |
| Moving average | **0.0318** | **0.1287** | **0.1783** |
| Gaussian | 0.0320 | 0.1290 | 0.1789 |
| Threshold | 0.0321 | 0.1292 | 0.1792 |

### CNN vs CNN-LSTM

| Model | Loss (MSE) | MAE | RMSE |
|---|---|---|---|
| CNN (moving average) | **0.0318** | **0.1287** | **0.1783** |
| CNN-LSTM | 0.1770 | 0.3165 | 0.4207 |

### CNN Training Progression

| Epoch | Train Loss | Val Loss | Train MAE | Val MAE |
|---|---|---|---|---|
| 1 | 1.20 | 0.80 | 0.70 | 0.40 |
| 10 | 0.10 | 0.25 | 0.15 | 0.20 |
| 20 | 0.03 | 0.05 | 0.08 | 0.09 |
| 30 | 0.01 | 0.04 | 0.06 | 0.07 |

### CNN-LSTM Training Progression (overfitting)

| Epoch | Train Loss | Val Loss | Train MAE | Val MAE |
|---|---|---|---|---|
| 1 | 0.90 | 0.80 | 0.50 | 0.40 |
| 10 | 0.05 | 0.30 | 0.05 | 0.20 |
| 20 | 0.02 | 0.50 | 0.02 | 0.40 |
| 30 | 0.01 | 0.45 | 0.01 | 0.40 |

**Key findings:**

- The standard CNN outperforms CNN-LSTM by ~2.5× on MAE. The MFPC voxel representation combined with TimeDistributed convolution already captures sufficient temporal context without explicit recurrence.
- Moving average denoising yields marginal improvement (~0.5% MAE reduction). MFPC fusion largely subsumes simple denoising benefits.
- The CNN-LSTM shows clear overfitting from epoch ~5 onward (training MAE 0.01, validation MAE 0.40 at epoch 30). The dataset scale (~10 subjects × 27 actions) is insufficient to regularise a model with two stacked LSTM layers.

---

## Repository Structure

```
.
├── train_cnn.py              # Temporal CNN — training + evaluation
├── train_cnn_lstm.py         # CNN-LSTM hybrid — training + evaluation
├── visualize_cnn_results.py  # 3D visualisation of CNN predictions vs ground truth
├── visualize_groundtruth.py  # Ground truth skeleton visualisation
├── visualize_mmwave_frame.py # Single mmWave point cloud frame visualisation
├── requirements.txt
├── models/                   # Saved CNN-LSTM checkpoint (trained_model_lstm.h5)
├── models2/                  # Saved CNN checkpoint (trained_model.h5)
└── DB_Coursework/            # mini-MMFI dataset (S01–S10, A01–A27)
```

---

## Stack

Python · TensorFlow/Keras · NumPy · Matplotlib · scikit-learn

---

## What I'd Improve Next

- **Attention over time** — replace the LSTM with a transformer-style temporal attention module; better long-range dependencies, less prone to overfitting on small datasets
- **PointNet backbone** — process raw point clouds directly rather than voxelising, preserving spatial resolution lost in grid projection
- **Transfer learning** — pre-train on the full MM-Fi dataset (40 subjects) before fine-tuning on the mini split; directly addresses the data scale bottleneck that caused CNN-LSTM overfitting
- **Advanced regularisation** — higher dropout rates, L2 weight decay, and early stopping for the LSTM variant
- **Data augmentation** — random rotation, flip, and scaling of point clouds to artificially expand the training set
- **Real-time inference** — quantise and optimise for embedded deployment (Raspberry Pi, NVIDIA Jetson)

---

## Applications

Healthcare & fall detection · Automotive in-cabin sensing · Smart home presence monitoring · Sports biomechanics · Through-wall gesture recognition

---

## References

1. Yang et al., *MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing*, NeurIPS, 2023.
2. Zhongang Cai et al., *HuMMan: Multi-modal 4D Human Dataset for Versatile Sensing and Modeling*, ECCV, Springer, 2022. https://doi.org/10.1007/978-3-031-20071-7_33
3. Sizhe An, Umit Y. Ogras, *MARS: mmWave-based Assistive Rehabilitation System for Smart Healthcare*, ACM TECS, 2021. https://doi.org/10.1145/3477003
4. Sizhe An, Umit Y. Ogras, *Fast and scalable human pose estimation using mmWave point cloud*, DAC, ACM, 2022. https://doi.org/10.1145/3489517.3530522
5. C. Iovescu and S. Rao, *The fundamentals of millimeter wave sensors*, Texas Instruments White Paper SPYY005A, 2020.
6. S. Z. Gurbuz et al., *Recent Advances in mmWave-Radar-Based Sensing, Its Applications, and Its Future Prospects: A Comprehensive Review*, IEEE Sensors Journal, vol. 23, no. 21, pp. 25818–25846, 2023.
