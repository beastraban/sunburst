# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-26

### Added

- Initial public release on PyPI as `sunburst-bayes`
- arXiv preprint: [2601.19957](https://arxiv.org/abs/2601.19957)
- Four-module GPU-native pipeline:
  - **Module 0 (SingleWhip):** Intelligent preprocessing with Gaussian diagnostics and whitening
  - **Module 1 (CarryTiger):** GPU ray casting with wavelet-based mode detection and ChiSao exploration
  - **Module 2 (GreenDragon/CarryTiger):** Vectorized batched L-BFGS peak refinement
  - **Module 3 (BendTheBow):** Adaptive radial evidence via Laplace integration
- Automatic CPU fallback when CuPy/GPU is unavailable
- `compute_evidence()` single-call API
- Benchmark suite with head-to-head comparison against dynesty and UltraNest
- GPU profiler for scaling analysis across dimensions
- Failure mode benchmark for challenging distributions
- Streamlit GUI for interactive exploration (`pip install sunburst-bayes[gui]`)
- Validated through 1024 dimensions on Gaussian posteriors
- Sub-linear O(d^0.67) empirical wall-clock scaling on GPU
- 1269× speedup over dynesty at 64 dimensions

### Hardware tested

- NVIDIA RTX 3080 Laptop GPU (10 GB VRAM)
- CUDA 13.0, CuPy 13.x
- Windows 11, Python 3.11
