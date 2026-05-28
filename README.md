# EMMA: Extracting Multiple physical parameters from Multimodal Data

**CVPR 2026**

[Farhat Shaikh](https://scholar.google.com/citations?hl=en&user=mbAOSW0AAAAJ), [Ayan Banerjee](https://scholar.google.com/citations?user=UAlc7tEAAAAJ&hl=en), [Sandeep K. S. Gupta](https://scholar.google.com/citations?user=U9bcQkMAAAAJ&hl=en)

**IMPACT Lab, School of Computing & Augmented Intelligence (SCAI), Arizona State University**

[**Project page**](https://impactlabasu.github.io/EMMA-CVPR2026/) · [**Demo video**](https://youtu.be/Uo79pVlM6Rk)

---

## Overview

EMMA is a physics-informed multimodal framework that recovers all identifiable dynamical parameters of a system directly from raw video, audio, and image-based time-series observations. Unlike prior video-only approaches that struggle with occluded states, hidden actuation inputs, and assumptions about known initial conditions, EMMA performs joint inference of **explicit parameters**, **implicit dynamical components**, and **calibration invariants** within a unified continuous-time model.

The user supplies the parametric structure of the governing ODE; EMMA solves the inverse problem of recovering its parameters, along with any latent forcing and invariants, from multimodal observations.

## Key contributions

- **Multi-modal dynamical parameter extraction** from video, audio, and time-series reconstructed from visual charts.
- **Recovery under unobserved forcing inputs** by inferring latent actuation (e.g. wheel speed) from audio.
- **Estimation of implicit dynamics** associated with unmeasured physical effects (e.g. frictional drag).
- **Invariant calibration from raw video**, eliminating assumptions about known initial conditions or coordinate frames.
- **Extensive validation** on 100+ scenarios: Delfys benchmark (75 videos), real-world rover and quadrotor, and simulation charts.

## Architecture

<p align="center"><img src="docs/EMMA-arc.png" alt="EMMA architecture" width="780" /></p>

EMMA follows a three-step pipeline: **Sense · Learn · Verify**.

1. **Sense.** Video, audio, and chart images are converted into time-aligned signals through modality-specific pipelines.
2. **Learn.** A Liquid Time-Constant (LTC) network models the system's latent dynamics in continuous time.
3. **Verify.** A differentiable ODE solver simulates the recovered parameters and checks them against the observations under a physics-informed loss.

## Results

EMMA delivers accurate multi-parameter recovery across diverse physical systems. Full tables and ablations are in the [paper](docs/42612.pdf).

| System | Parameters recovered | EMMA error | Best baseline |
|--------|----------------------|------------|---------------|
| Pendulum (90 cm) | Length *L*, damping *τ* | **L = 0.86 ± 0.07 m** (GT 0.90) | Delfys, PySINDy |
| Torricelli (med.) | Drainage *k* | **0.0132 ± 0.0008** (GT 0.0128) | matches Delfys |
| Sliding block (med.) | Angle *α*, friction *μ* | **α = 24.72°, μ = 0.205** (GT 25°, 0.20) | Delfys, PySINDy |
| LED decay (med.) | γ | **0.91 ± 0.0** (GT 0.92) | matches Delfys |
| Rover | 9 params (5 with known ground truth) | **8.8 % ± 1.7 %** mean error | *first work under hidden forcing* |
| Quadrotor | 12 params (7 with known ground truth) | **15.9 % ± 7.4 %** mean error | *first work under hidden forcing* |
| Simulation charts | Lotka-Volterra, Lorenz, F8 Crusader, HIV, AID | **>10× lower error** than PySINDy on implicit dynamics | PySINDy |

Compared against **PAIG**, **NIRPI**, and **Delfys** on the video benchmarks and **PySINDy** on the chart-based simulations.

## Supported systems

| Category | Systems |
|----------|---------|
| Delfys benchmark | Pendulum, Torricelli drainage, Sliding block, LED decay, Free fall |
| Real-world platforms | Differential-drive rover (9 params), 6-DoF quadrotor (12 params) |
| Simulation charts | Lotka-Volterra, Chaotic Lorenz, F8 Crusader, HIV therapy, AID (Type-1 diabetes) |

## Installation

Tested with **Python 3.10+** on macOS and Linux.

```bash
git clone https://github.com/ImpactLabASU/EMMA-CVPR2026.git
cd EMMA-CVPR2026
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

**System tools**

- [FFmpeg](https://ffmpeg.org/) on your `PATH` (MoviePy uses it for audio extraction): `brew install ffmpeg` (macOS) or `sudo apt install ffmpeg` (Ubuntu).
- YOLO weights (default `yolo11m.pt`): `pip install ultralytics` then `yolo download model=yolo11m.pt`, or download from the Ultralytics releases page.
- A CUDA GPU is optional; every script falls back to CPU automatically.

## Repository layout

| Folder | Purpose | Entry points |
| --- | --- | --- |
| `Baseline/` | Physics-informed EMMA pipelines (Free Fall, LED, Pendulum, Sliding Block, Torricelli) plus ablation utilities. | `FreeFall/free_fall.py`, `LED/led.py`, `Pendulum/run-*.py`, `Sliding block/sliding_block*.py`, `Torricelli/toricelli*.py`, `architecture_ablation.py`, `run_additional_ablations.py` |
| `Rover/` | Rover perception, parameter estimation, multimodal ablations, helper shell script. | `run.py`, `rover-ablation.py`, `rover_multimodal_ablation.py`, `run_rover_ablation.sh` |
| `Drone/` | Drone pipeline orchestrator (vision + audio + EMMA optimization). | `new_run.py` |
| `CGM/` | Continuous glucose monitor chart digitizer. | `extract_cgm_data.py` |

## Data

- **Baseline datasets** come from the Delfys "Physical Parameter Prediction" set on Kaggle (https://www.kaggle.com/datasets/jaswar/physical-parameter-prediction). Download it and copy the experiment folders into `Baseline/`; the scripts discover the data automatically.
- **Sample rover and drone videos** are available here: **[Dropbox](https://www.dropbox.com/scl/fo/cjiym1h53puvv2ml6o8vn/APkfhTz64DnkYkHt554ZPj0?rlkey=hw3odtpzn6vl2nsfbe4pkekcq&dl=0)**. Place them under `Rover/` and `Drone/`.

## Usage

### Baseline pipelines

Each baseline follows the same recipe:

1. `cd Baseline/<Experiment>/`
2. Edit the configuration block inside `main()`:
   - `video_path`: path to the source video; leave empty to reuse existing data files.
   - `weights_path`: YOLO weights (`yolo11m.pt` by default).
   - `pixel_to_meter` (Free Fall, Torricelli, Sliding Block): set from your calibration grid.
   - `output_folder`: a unique run directory (e.g. `run_01`); the script creates `output/` and `data/` under it.
3. Run `python3 <script>.py`.
4. Optional: `python3 <script>.py --simulation-only` skips retraining and reuses the latest `*_coefficients.csv` and `*_emma_final_model.pth` (Free Fall, LED, Pendulum).

| Experiment | Script | Key outputs |
| --- | --- | --- |
| Free Fall | `FreeFall/free_fall.py` (`free_fall-m.py` for the medium set) | trajectory CSV, `free_fall_coefficients.csv`, trained model, annotated video |
| LED decay | `LED/led.py` | trajectory CSV, `led_coefficients.csv`, trained model, intensity figures |
| Pendulum | `Pendulum/run-45.py`, `run-90.py`, `run-150.py` | `thetaData.txt`, `omegaData.txt`, `pendulum_coefficients.csv`, trained model |
| Sliding block | `Sliding block/sliding_block.py` (`-low`, `-med` variants) | trajectory CSVs, `sliding_block_coefficients.csv`, trained model |
| Torricelli | `Torricelli/toricelli.py` (`toricelli-m.py`, `torricelli-sm.py`) | height trajectories, `torricelli_coefficients.csv`, trained model |

**PySINDy baselines.** Each experiment folder has `pysindy_results/pysindy.py`; run it from that folder (after the main pipeline has written the EMMA-formatted CSVs) for sparse-regression baselines.

**Ablations.** From `Baseline/`: `python3 architecture_ablation.py` and `python3 run_additional_ablations.py` (require pendulum datasets under `Baseline/Pendulum-EMMA/<angle>_v*/data/`).

### Rover

```bash
cd Rover
# set video_path and weights_path in run.py (see the CONFIGURATION SECTION)
python3 run.py
```

Outputs: `rover_coefficients.csv`, `rover_EMMA_final_model.pth`, plots, GIF. Ablations: `python3 rover-ablation.py`, `python3 rover_multimodal_ablation.py`, or `bash run_rover_ablation.sh` (edit variables first). If you already have processed `data/*.txt`, set `video_path = ""` to skip detection.

### Drone

```bash
cd Drone
EMMA_RUN_ORCHESTRATOR=1 python3 new_run.py --video /path/to/DroneVideo.mp4 --weights /path/to/yolo11m.pt
```

> **Note:** Full orchestration also needs an external `Dronepipeline/` folder containing `droneExtract.py`, `droneExtractAudio.py`, and `EMMA_drone_torch_ltc_optimized.py`. These are not bundled here; without them, `new_run.py` falls back to the local vision-only pipeline.

### CGM chart digitizer

```bash
cd CGM
python3 extract_cgm_data.py   # reads CGMData.png, writes cgm_data.txt + a visualization
```

## Troubleshooting

- **Module not found:** re-run `pip install -r requirements.txt` in the active virtual environment. For `torch`/`torchvision`, use the [PyTorch selector](https://pytorch.org/get-started/locally/).
- **YOLO weights missing:** download `yolo11m.pt` and point `weights_path` to it.
- **FFmpeg errors:** install FFmpeg (`brew install ffmpeg` / `sudo apt install ffmpeg`).

## Citation

```bibtex
@InProceedings{Shaikh_2026_CVPR,
    author    = {Shaikh, Farhat and Banerjee, Ayan and Gupta, Sandeep},
    title     = {EMMA: Extracting Multiple physical parameters from Multimodal Data},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {1716-1725}
}
```

Also on [arXiv](https://arxiv.org/abs/2605.24047).
