# SunBURST

[![arXiv](https://img.shields.io/badge/arXiv-2601.19957-b31b1b.svg)](https://arxiv.org/abs/2601.19957)
[![PyPI version](https://badge.fury.io/py/sunburst-bayes.svg)](https://badge.fury.io/py/sunburst-bayes)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/beastraban/sunburst/actions/workflows/ci.yml/badge.svg)](https://github.com/beastraban/sunburst/actions/workflows/ci.yml)

**Seeded Universe Navigation — Bayesian Unification via Radial Shooting Techniques**

A GPU-accelerated Bayesian evidence calculator achieving machine-precision accuracy through 1024 dimensions with sub-linear scaling.

## Features

- **Extreme scalability**: O(D^0.67) scaling vs O(exp(D)) for traditional methods
- **GPU acceleration**: 1000× speedup over dynesty/PolyChord at matched dimensions
- **High dimensions**: Works reliably from 2D to 1024D+
- **Pure Python**: No compilation required, NumPy/CuPy compatible
- **Automatic mode detection**: Handles multimodal posteriors automatically

## Installation

```bash
pip install sunburst-bayes
```

For GPU acceleration (optional but recommended):
```bash
pip install cupy-cuda11x  # For CUDA 11.x
# or
pip install cupy-cuda12x  # For CUDA 12.x
```

## Quick Start

```python
from sunburst import compute_evidence, get_array_module

# Define your log-likelihood (GPU-native, handles batched inputs)
def log_likelihood(x):
    xp = get_array_module(x)  # CuPy if GPU, NumPy if CPU
    return -0.5 * xp.sum(x**2, axis=1)  # Gaussian

# Define parameter bounds
dim = 64
bounds = [(-10, 10)] * dim

# Compute evidence
result = compute_evidence(log_likelihood, bounds)

print(f"log Z = {result.log_evidence:.4f} ± {result.log_evidence_std:.4f}")
print(f"Peaks found: {result.n_peaks}")
print(f"Time: {result.wall_time:.2f}s")
```

Expected output (RTX 3080, 64D Gaussian):
```
log Z = -91.8939 ± 0.0001
Peaks found: 1
Time: 0.71s
```

The true value is `log Z = -91.8939` for a 64D unit Gaussian on [-10, 10]^64. SunBURST recovers it to machine precision in under a second.

## When Not to Use SunBURST

SunBURST works best on posteriors that are approximately Gaussian near their peaks. It is **not the right tool** when:

- **Heavy-tailed posteriors** (e.g. Student-t with low ν). The Laplace approximation underestimates probability mass in the tails. Errors can exceed 100%.
- **Highly curved or banana-shaped** posteriors where the Hessian at the peak misrepresents the global geometry.
- **Ring/shell distributions** (e.g. donut posteriors) where probability mass concentrates far from any peak.
- **You need posterior samples**, not evidence. SunBURST computes log Z; it does not produce MCMC-like samples. Use dynesty or PolyChord if you need the posterior itself.

For these cases, traditional nested samplers remain more reliable. See Table 4 in the [paper](https://arxiv.org/abs/2601.19957) for detailed failure-mode benchmarks.

## Interactive GUI

An interactive Streamlit demo is available:

```bash
git clone https://github.com/beastraban/sunburst.git
cd sunburst/sunburst_super_gui
pip install streamlit
streamlit run app.py
```

## Performance Benchmarks

Tested on RTX 3080 Laptop GPU with `n_oscillations=1`:

| Dimension | SunBURST | dynesty | UltraNest | Speedup |
|-----------|----------|---------|-----------|---------|
| 2D        | 0.39s    | 0.61s   | 0.87s     | 1.6–2.2× |
| 8D        | 0.42s    | 37s     | 54s       | 88–129× |
| 64D       | 0.71s    | TIMEOUT | TIMEOUT   | >1200× |
| 256D      | 2.72s    | —       | —         | ∞ |
| 1024D     | 14.0s    | —       | —         | ∞ |

**TIMEOUT** = >600s (10 minutes)


![SunBURST Scaling](assets/Scaling_figure.png)

*SunBURST completes in seconds where traditional methods timeout.*

## Built-in Test

Verify your installation:

```python
import sunburst
result = sunburst.test(dim=64)  # Runs Gaussian benchmark
```

Or from command line:
```bash
sunburst --test gaussian --dim 64
```

## Configuration Options

```python
result = compute_evidence(
    log_likelihood,
    bounds,
    n_oscillations=1,     # 1=fast, 3=conservative mode detection
    fast=True,            # Fast Hessian estimation
    return_peaks=True,    # Include peak locations in result
    verbose=False,        # Suppress progress output
    seed=42,              # Reproducibility
)
```

## Result Object

```python
result.log_evidence        # float: Estimated log Z
result.log_evidence_std    # float: Uncertainty estimate
result.n_peaks             # int: Number of modes found
result.peaks               # ndarray: (n_peaks, dim) peak locations
result.hessians            # list: Hessian matrices at peaks
result.log_evidence_per_peak  # ndarray: Evidence contribution per peak
result.wall_time           # float: Total computation time
result.module_times        # dict: Per-stage timing breakdown
result.n_likelihood_calls  # int: Total likelihood evaluations
result.config              # dict: Configuration used
```

## GPU Utilities

```python
from sunburst import gpu_available, gpu_info, get_array_module

if gpu_available():
    print(gpu_info())
    xp = get_array_module()  # Returns cupy if available, else numpy
```

## Architecture

SunBURST uses a 4-stage pipeline, named after Guang Ping Yang Style Tai Chi forms:

1. **CarryTiger (抱虎歸山)**: Mode detection via ray casting
2. **GreenDragon (青龍出水)**: Peak refinement with L-BFGS
3. **BendTheBow (彎弓射虎)**: Evidence calculation via Laplace approximation
4. **GraspBirdsTail (攬雀尾)**: Optional dimensional reduction

## Citation

If you use SunBURST in your research, please cite:

```bibtex
@article{wolfson2026sunburst,
    title={SunBURST: Deterministic GPU-Accelerated Bayesian Evidence 
           via Mode-Centric Laplace Integration},
    author={Wolfson, Ira},
    journal={arXiv preprint arXiv:2601.19957},
    year={2026}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please see our [contributing guidelines](CONTRIBUTING.md).

## Acknowledgments

Module names honor Master Donald Rubbo and the Guang Ping Yang Style Tai Chi (廣平楊式太極拳) tradition.
1