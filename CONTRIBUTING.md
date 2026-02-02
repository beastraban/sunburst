# Contributing to SunBURST

Thank you for your interest in contributing to SunBURST! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/sunburst.git
   cd sunburst
   ```
3. **Install in development mode:**
   ```bash
   pip install -e .
   ```
4. **Install test dependencies:**
   ```bash
   pip install dynesty ultranest matplotlib
   ```

## Running Tests

Before submitting any changes, verify that existing tests pass:

```bash
# Quick sanity check (< 1 minute)
cd benchmarks
python benchmark_suite.py --quick

# GPU profiling (requires NVIDIA GPU + CuPy)
python gpu_profiler.py --dims 2,8,32 --runs 2

# Failure mode tests
python failure_mode_benchmark.py --dims 2,4,8
```

## Code Style

- Follow PEP 8 conventions
- Use descriptive variable names
- All GPU code must handle the CPU fallback path (`cupy` unavailable)
- Use `get_array_module(x)` to maintain GPU/CPU compatibility
- Log-space arithmetic for numerical stability (no raw probabilities at high dimensions)
- **No silent exception handling** — always log errors with messages:
  ```python
  # WRONG
  try:
      result = compute_something()
  except:
      pass

  # CORRECT
  try:
      result = compute_something()
  except Exception as e:
      print(f"WARNING: compute_something failed: {e}")
      result = fallback_value
  ```

## Adding Benchmark Functions

To add a new test function to the benchmark suite:

1. Add a class in `benchmarks/benchmark_suite.py` inheriting from `TestFunction`
2. Implement `log_likelihood(self, x)` and `log_evidence_true(self)` (if analytical)
3. Register the class in the `func_map` dictionary in `run_benchmark()`
4. Test at dimensions 2, 8, and 32 before submitting

## Submitting Changes

### Pull Request Checklist

- [ ] Code follows the style guidelines above
- [ ] GPU/CPU compatibility maintained (no CuPy-only paths without fallback)
- [ ] Existing benchmark results are not broken
- [ ] New features include at least a minimal test
- [ ] Commit messages are descriptive

### PR Process

1. Create a feature branch: `git checkout -b feature/my-improvement`
2. Make your changes with clear, atomic commits
3. Push to your fork: `git push origin feature/my-improvement`
4. Open a Pull Request against `main`
5. Describe what changed and why

## Reporting Bugs

Open a GitHub issue with:

- **Description:** What went wrong
- **Environment:** Python version, OS, GPU model (if applicable), CuPy/CUDA version
- **Minimal reproducer:** Smallest code that triggers the bug
- **Expected vs actual output**

## Feature Requests

Open a GitHub issue tagged `[Feature Request]` describing:

- The use case or problem you're trying to solve
- Your proposed approach (if any)
- Whether you'd be willing to contribute the implementation

## Questions

For questions about the algorithm or usage, open a GitHub Discussion or issue tagged `[Question]`.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
