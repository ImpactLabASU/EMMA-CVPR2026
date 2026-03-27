
# EMMA: Extracting Multiple physical parameters from Multimodal Data

### CVPR 2026 (Main Conference)

[Farhat Shaikh](mailto:fshaik12@asu.edu), [Ayan Banerjee](mailto:abanerj3@asu.edu), [Sandeep Gupta](mailto:Sandeep.gupta@asu.edu)

**IMPACT Lab, School of Computing & Augmented Intelligence (SCAI), Arizona State University**


---

## Overview

EMMA is a physics-informed multimodal framework that recovers all identifiable dynamical parameters of a system directly from raw video, audio, and image-based time-series observations. Unlike prior video-only approaches, EMMA performs joint inference of explicit parameters, implicit dynamical components, and calibration invariants within a unified continuous-time model.


## Key Contributions

- **Multi-modal dynamical parameter extraction** from video, audio, and time-series reconstructed from visual charts
- **Recovery under unobserved forcing inputs** by inferring latent actuation inputs (e.g., wheel speed) from audio
- **Estimation of implicit dynamics** associated with unmeasured physical effects (e.g., frictional drag)
- **Invariant calibration from raw video** eliminating assumptions about known initial conditions or coordinate frames
- **Extensive validation** on 100+ scenarios: Delfys benchmark (75 videos), real-world rover/drone, and simulation charts

## Architecture


EMMA consists of three stages:
1. **Unified multi-modal feature extraction** from video, audio, and images
2. **Liquid Time-Constant (LTC) network** modeling continuous-time latent dynamics
3. **Multi-parameter estimation** via physics-constrained optimization with differentiable simulation

## Results

EMMA delivers accurate multi-parameter recovery across diverse physical systems:

| System | Parameters | EMMA Error | Best Baseline Error |
|--------|-----------|------------|-------------------|
| Pendulum | Length, Damping | 4.8% avg | 12.6% (PySINDy) |
| Rover (9 params) | Geometry, Mass, Friction | 14.5% avg | N/A (first work) |
| Drone (12 params) | Thrust, Torque, Geometry | 16.1% avg | N/A (first work) |

## Installation

```bash
git clone https://github.com/Faruu18/EMMA.git
cd EMMA
pip install -r requirements.txt
```

### Dependencies

- Python >= 3.9
- PyTorch >= 2.0
- ncps (Liquid Time-Constant networks)
- YOLOv11 (ultralytics)
- OpenCV
- librosa
- MoviePy
- scipy, numpy, matplotlib



## Supported Systems

| Category | Systems |
|----------|---------|
| Benchmark (Delfys) | Pendulum, Torricelli Drainage, Sliding Block, LED Decay, Free Fall |
| Real-world | Differential-drive Rover (9 params), 6-DoF Quadrotor (12 params) |
| Simulation Charts | Lotka-Volterra, Chaotic Lorenz, F8 Cruiser, HIV Therapy, AID (Insulin Delivery) |

## Citation

If you find this work useful, please cite:

```bibtex

```

## Acknowledgments

This work was supported in part by NSF FDTBiotech and DARPA FIRE.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
