# Reproducing the Paper Results

This document describes how to reproduce the benchmark results reported in:

> **SunBURST: Deterministic GPU-Accelerated Bayesian Evidence via Mode-Centric Laplace Integration**
> Ira Wolfson, 2026. arXiv:2601.19957

## Prerequisites

**Hardware used in the paper:**
- GPU: NVIDIA RTX 3080 Laptop (10 GB VRAM)
- CUDA: 13.0
- OS: Windows 11
- Python: 3.11

**Minimum requirements:**
- Any NVIDIA GPU with ≥4 GB VRAM (for dimensions ≤256)
- ≥8 GB VRAM recommended for 512D–1024D
- CPU-only mode works but timings will differ from those reported

**Software setup:**
```bash
# Install SunBURST
pip install sunburst-bayes

# Or from source
git clone https://github.com/beastraban/sunburst.git
cd sunburst
pip install -e .

# Install competitor methods for comparison benchmarks
pip install dynesty ultranest

# Install profiling extras
pip install matplotlib
```

## Repository Layout

```
benchmarks/
├── benchmark_suite.py          # Head-to-head comparison (Tables 1–2)
├── gpu_profiler.py             # Scaling analysis (Table 3, Figures)
├── failure_mode_benchmark.py   # Failure mode tests (Table 4)
├── header_utils.py             # Standardized output headers
├── README.md                   # Benchmark documentation
└── data/                       # Raw results from publication runs
    ├── *.csv                   # Timing and accuracy data
    └── *.xlsx                  # Aggregated statistics
```

## Reproducing Each Result

### Table 1: SunBURST vs Competitors (Accuracy)

```bash
cd benchmarks
python benchmark_suite.py --full \
    --methods sunburst,dynesty,ultranest \
    --functions gaussian,correlated,mixture,cigar
```

This runs all methods across dimensions 2–256 on four test functions. Results are saved to `results/benchmark_results_TIMESTAMP.csv`.

**Expected runtime:** 2–4 hours (dominated by dynesty/UltraNest at higher dimensions). SunBURST completes its portion in under 1 minute total.

**Note:** Competitor methods (dynesty, UltraNest) are stochastic — expect small variations between runs. SunBURST is deterministic and should produce identical results given the same hardware.

### Table 2: Wall-Clock Speedups

Speedup ratios are computed from the same benchmark_suite.py run above. The summary table printed at the end shows SunBURST time vs dynesty time per (function, dimension) pair.

**Timeout behavior:** Competitors that exceed 600 seconds (10 minutes) per run are marked as timed out. At 16D+ on our hardware, dynesty and UltraNest consistently time out.

### Table 3: GPU Scaling Analysis

```bash
cd benchmarks
python gpu_profiler.py \
    --dims 2,4,8,16,32,64,128,256,512,1024 \
    --runs 4 \
    --n-oscillations 1 \
    --fast
```

Outputs to `profiler_results/`:
- `raw_results_TIMESTAMP.csv` — per-run timing data
- `summary_TIMESTAMP.csv` — per-dimension aggregated statistics
- `dashboard.png` — scaling and accuracy plots
- `session_TIMESTAMP.log` — session log with hardware info

**Expected runtime:** ~5 minutes on RTX 3080.

### Table 4: Failure Mode Analysis

```bash
cd benchmarks
python failure_mode_benchmark.py \
    --dims 2,4,8,16 \
    --functions student_t,mixture,banana,cigar \
    --n-oscillations 3
```

Outputs to `failure_mode_results/failure_mode_results_TIMESTAMP.csv`.

**Note:** Higher `--n-oscillations` improves robustness on hard problems at the cost of runtime.

### Quick Validation (< 1 minute)

To verify the installation works correctly without running the full suite:

```bash
cd benchmarks
python benchmark_suite.py --quick
```

This runs SunBURST, dynesty, and UltraNest on dimensions 2, 8, and 32 only.

## Pre-computed Results

The `benchmarks/data/` directory contains the raw CSV and Excel files from our publication runs on the RTX 3080 Laptop GPU. These can be used to verify figures and tables without re-running benchmarks.

## Differences You May See

- **Different GPU:** Wall-clock times scale roughly with GPU compute throughput. Accuracy should be identical for SunBURST (deterministic). A100/H100 users will see faster times; older GPUs will be slower.
- **CPU-only mode:** SunBURST automatically falls back to NumPy/SciPy. Evidence values will match but timings will be significantly slower (no GPU parallelism).
- **Competitor variation:** dynesty and UltraNest are stochastic samplers. Evidence estimates will vary by ~1–5% between runs; wall-clock times vary by ~10–20%.
- **CUDA version:** Minor timing differences may occur across CUDA toolkit versions. Evidence values should be unaffected.

## Contact

For questions about reproducibility, open a GitHub issue or contact the author.
