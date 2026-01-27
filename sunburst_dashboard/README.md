# SunBURST Dashboard 🌞

Real-time local monitoring for SunBURST Bayesian evidence calculations.

Like TensorBoard, but for Bayesian inference.

## Features

- **Live Pipeline Status**: Watch CarryTiger → GreenDragon → BendTheBow progress
- **2D Visualization**: See ray samples and detected peaks in real-time
- **Run History**: Track all your evidence calculations in one place
- **Fully Local**: No telemetry, no cloud - everything runs on your machine

## Quick Start

### 1. Install dependencies

```bash
pip install websockets
```

### 2. Start the dashboard server

```bash
python server.py
```

This opens `http://localhost:8080` in your browser automatically.

### 3. Run SunBURST with dashboard callbacks

```python
from sunburst_dashboard.server import DashboardClient

# Create dashboard client
dashboard = DashboardClient()

# Use callbacks in your pipeline
dashboard.on_pipeline_start(dim=64, bounds=bounds)
dashboard.on_module_start("CarryTiger")
# ... your computation ...
dashboard.on_peaks_found(peaks, likelihoods)
dashboard.on_module_end("CarryTiger", time_seconds=1.2)
# ... etc ...
dashboard.on_result(log_evidence=-92.31, n_peaks=3, time_seconds=2.1)
```

## API Reference

### DashboardClient Methods

| Method | Description |
|--------|-------------|
| `on_pipeline_start(dim, bounds, config)` | Called when pipeline starts |
| `on_pipeline_end(success)` | Called when pipeline ends |
| `on_module_start(module_name)` | Called when a module starts |
| `on_module_end(module_name, time_seconds, **extra)` | Called when a module ends |
| `on_progress(percent, label, evals)` | Progress updates |
| `on_rays_update(rays_cast, total_rays, ray_samples)` | Ray casting updates |
| `on_peaks_found(peaks, likelihoods, widths)` | Peak detection updates |
| `on_result(log_evidence, n_peaks, time_seconds)` | Final result |
| `on_error(error_message, module)` | Error reporting |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Your Machine                                               │
│                                                             │
│  ┌─────────────────┐    WebSocket     ┌────────────────┐   │
│  │   SunBURST      │ ───────────────► │  server.py     │   │
│  │   Pipeline      │   localhost:8765  │  (Python)      │   │
│  │   + callbacks   │                  └───────┬────────┘   │
│  └─────────────────┘                          │            │
│                                       localhost:8080       │
│                                               │            │
│                                       ┌───────▼────────┐   │
│                                       │   Browser      │   │
│                                       │   dashboard.html│   │
│                                       └────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `server.py` | WebSocket server + DashboardClient |
| `dashboard.html` | Browser UI (single file, no build step) |
| `example_with_dashboard.py` | Demo showing how to use callbacks |

## Integrating with Your Pipeline

Add callbacks to your existing SunBURST code:

```python
# In your pipeline's compute_evidence method:

def compute_evidence(self, log_L, bounds, callbacks=None):
    if callbacks:
        callbacks.on_pipeline_start(dim=len(bounds), bounds=bounds)
    
    # CarryTiger
    if callbacks:
        callbacks.on_module_start("CarryTiger")
    
    peaks = self._carry_tiger(log_L, bounds)
    
    if callbacks:
        callbacks.on_peaks_found(peaks, ...)
        callbacks.on_module_end("CarryTiger", time_seconds=elapsed)
    
    # ... etc ...
```

## License

MIT - Same as SunBURST
