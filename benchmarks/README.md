# SunBURST Benchmarks

This folder contains benchmark scripts and results comparing SunBURST against traditional nested sampling methods.

## Quick Start

```bash
# Install sunburst (from repo root)
pip install -e .

# Install competitor methods (optional)
pip install dynesty ultranest

# Run quick benchmark
cd benchmarks
python benchmark_suite.py --quick

# Run full benchmark
python benchmark_suite.py --full
```

## Benchmark Scripts

| Script | Description |
|--------|-------------|
| `benchmark_suite.py` | Main comparison vs dynesty/UltraNest |
| `gpu_profiler.py` | GPU performance profiling across dimensions |
| `failure_mode_benchmark.py` | Tests on challenging distributions |
| `header_utils.py` | Utilities for standardized output headers |

## Command Line Options

### benchmark_suite.py

```
--quick              Quick test (2D, 8D, 32D only)
--full               Full suite (2D through 256D)
--dims 2,8,32,64     Custom dimensions
--methods sunburst,dynesty,ultranest
--functions gaussian,correlated,mixture,cigar,rosenbrock
--output ./results   Output directory for CSV files
```

### gpu_profiler.py

```
--dims 2,4,8,16,...  Dimensions to profile
--runs 4             Runs per dimension
--n-oscillations 1   SunBURST n_oscillations parameter
--fast               Use fast Hessian estimation
--output-dir ./profiler_results
```

### failure_mode_benchmark.py

```
--dims 2,4,8         Dimensions to test
--functions student_t,mixture,banana,cigar,donut
--n-oscillations 3   Higher for robustness on hard problems
--output-dir ./failure_mode_results
```

## Test Functions

| Function | Description | Evidence |
|----------|-------------|----------|
| gaussian | Isotropic Gaussian | Analytical |
| correlated | Correlated Gaussian | Analytical |
| cigar | Axis-aligned anisotropic | Analytical |
| mixture | Multimodal (4 Gaussians) | Analytical |
| rosenbrock | Curved valley | Numerical |

## Results

Benchmark results are saved to `results/benchmark_results_TIMESTAMP.csv` with columns:
- method, function, dim
- log_evidence, log_evidence_true, error_percent
- wall_time, n_calls, status

## Data Files

The `data/` subfolder contains raw benchmark results from our publication runs:
- CSV files with timing and accuracy data
- Excel summaries with aggregated statistics

## Reproducing Paper Results

To reproduce the results from the paper:

```bash
python benchmark_suite.py --full --methods sunburst,dynesty,ultranest --functions gaussian,correlated,mixture,cigar
```

Note: Full benchmarks can take several hours depending on hardware. Use `--quick` for rapid testing.
