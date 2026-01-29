# ☀️ SunBURST Super-GUI

The ultimate Bayesian evidence calculator interface - combining real-time monitoring with full analysis tools.

## Features

### 🚀 Run Analysis
- Built-in test functions (Gaussian, Bimodal, Trimodal, Rosenbrock, Eggbox, Cigar)
- Configurable dimensions (2D - 128D)
- Real-time pipeline progress
- Immediate results with error estimation

### 📊 Corner Plot
- Generate corner plots from Laplace approximation
- No MCMC sampling needed - analytical marginals!
- Interactive Plotly visualization

### ⚖️ Model Comparison
- Compare multiple models side-by-side
- Bayes factor interpretation
- Posterior model probabilities

### 📈 Scaling Analysis
- Track timing across dimensions
- Automatic power-law fitting
- Visualize O(d^α) scaling

### 📜 Run History
- Complete log of all runs
- Export to CSV

### 🔴 Real-time Dashboard
- WebSocket-based live monitoring
- See CarryTiger → GreenDragon → BendTheBow progress
- 2D projection of peaks and rays

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Streamlit GUI (Full Analysis)

```bash
streamlit run app.py
```

Opens at http://localhost:8501

### Option 2: Real-time Dashboard (Live Monitoring)

Terminal 1:
```bash
python dashboard_server.py
```

Opens at http://localhost:8080

Terminal 2:
```python
from dashboard_server import DashboardClient

dashboard = DashboardClient()
dashboard.on_pipeline_start(dim=64, bounds=bounds)
# ... your SunBURST run with callbacks ...
dashboard.on_result(log_evidence=-92.31, n_peaks=3, time_seconds=2.1)
```

## Screenshots

### Run Analysis Tab
```
┌─────────────────────────────────────────────────────────────┐
│  Configuration          │  Results                          │
│  ──────────────────     │  ──────────────────────────────── │
│  Function: Gaussian     │  log Z    Modes    Time    Error  │
│  Dimension: 64          │  -92.31   2        0.89s   0.02%  │
│  σ: 1.0                 │                                    │
│  Bounds: ±10            │  [Likelihood Surface Plot]        │
│                         │  [Timing Breakdown]               │
│  [▶️ Run SunBURST]       │  [Parameter Estimates Table]      │
└─────────────────────────────────────────────────────────────┘
```

### Corner Plot Tab
```
┌────────┬────────┬────────┐
│   θ₁   │        │        │
│  ▁▂▅█▅▂│        │        │
├────────┼────────┼────────┤
│ ░░▓▓░░ │   θ₂   │        │
│ ░▓██▓░ │  ▁▅█▅▁ │        │
├────────┼────────┼────────┤
│ ░░▓▓░░ │ ░░▓░░  │   θ₃   │
│ ░▓██▓░ │ ░▓█▓░  │  ▂▅█▅▂ │
└────────┴────────┴────────┘
```

### Model Comparison Tab
```
Model         log Z      BF Interpretation
────────────  ────────   ─────────────────
ΛCDM_8D       -45.23     Reference
wCDM_8D       -48.91     Strong evidence for ΛCDM
```

## Files

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application |
| `dashboard_server.py` | WebSocket server for real-time monitoring |
| `dashboard.html` | Browser UI for real-time dashboard |
| `requirements.txt` | Python dependencies |

## Integrating with Your SunBURST Pipeline

### For Streamlit GUI

Edit `app.py` and update the import:

```python
# Replace this:
HAS_SUNBURST = False

# With:
from SUNBURST_PIPELINE_v1_7 import SunburstPipeline, PipelineConfig
HAS_SUNBURST = True
```

### For Real-time Dashboard

Add callbacks to your pipeline:

```python
from dashboard_server import DashboardClient

def compute_evidence_with_dashboard(log_L, bounds):
    dashboard = DashboardClient()
    
    dashboard.on_pipeline_start(dim=len(bounds), bounds=bounds)
    
    # CarryTiger
    dashboard.on_module_start("CarryTiger")
    peaks = carry_tiger(log_L, bounds)
    dashboard.on_peaks_found(peaks, likelihoods)
    dashboard.on_module_end("CarryTiger", time_seconds=t1)
    
    # GreenDragon
    dashboard.on_module_start("GreenDragon")
    refined_peaks = green_dragon(peaks)
    dashboard.on_module_end("GreenDragon", time_seconds=t2)
    
    # BendTheBow
    dashboard.on_module_start("BendTheBow")
    log_Z = bend_the_bow(refined_peaks)
    dashboard.on_module_end("BendTheBow", time_seconds=t3)
    
    dashboard.on_result(log_evidence=log_Z, n_peaks=len(peaks), time_seconds=total)
    
    return log_Z
```

## Test Functions

| Function | Description | Analytical Evidence |
|----------|-------------|---------------------|
| Gaussian | Isotropic Gaussian | ✅ |
| Bimodal | Two equal modes | ✅ |
| Trimodal | Three modes (triangle) | ✅ |
| Rosenbrock | Banana-shaped | ❌ |
| Eggbox | Periodic multimodal | ❌ |
| Cigar | Anisotropic (elongated) | ✅ |

## Tips

1. **Start simple**: Use Gaussian at 8D to validate accuracy
2. **Test multimodality**: Bimodal checks mode detection
3. **Scale up gradually**: 8 → 16 → 32 → 64 → 128D
4. **Compare models**: Add multiple runs to comparison tab
5. **Check scaling**: Use scaling tab to verify O(d^0.67)

## Requirements

- Python 3.8+
- CUDA-capable GPU (for actual SunBURST)
- Modern browser (for dashboard)

## License

MIT License - Braude College of Engineering

---

*☀️ SunBURST: Seeded Universe Navigation — Bayesian Unification via Radial Shooting Techniques*
