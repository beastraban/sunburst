#!/usr/bin/env python3
"""
SunBURST Benchmark Header Utilities
===================================

Provides standardized header generation for all benchmark outputs.
Ensures consistent configuration documentation across all test suites.

Usage:
    from header_utils import generate_header, get_system_info
    
    header = generate_header(
        test_name="GPU Profiler",
        version="3.0",
        config={
            'n_oscillations': 1,
            'fast': True,
            'runs_per_dimension': 4,
            'dimensions': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        }
    )

Author: Ira Wolfson
Date: January 2026
"""

import platform
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# =============================================================================
# SUNBURST VERSION
# =============================================================================

try:
    from sunburst import __version__ as SUNBURST_VERSION
except ImportError:
    SUNBURST_VERSION = "not installed"

# =============================================================================
# SYSTEM INFO
# =============================================================================

def get_cpu_info() -> Dict[str, Any]:
    """Get CPU information."""
    info = {
        'processor': platform.processor() or 'Unknown',
        'architecture': platform.machine(),
        'physical_cores': None,
        'logical_cores': os.cpu_count(),
        'cpu_name': 'Unknown',
    }
    
    # Try to get more detailed CPU info
    try:
        import psutil
        info['physical_cores'] = psutil.cpu_count(logical=False)
        info['logical_cores'] = psutil.cpu_count(logical=True)
    except ImportError:
        pass
    
    # Try to get CPU name on Windows
    if sys.platform == 'win32':
        try:
            import subprocess
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'name'],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                info['cpu_name'] = lines[1].strip()
        except Exception:
            pass
    # Try to get CPU name on Linux
    elif sys.platform == 'linux':
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('model name'):
                        info['cpu_name'] = line.split(':')[1].strip()
                        break
        except Exception:
            pass
    
    return info


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information."""
    info = {
        'available': False,
        'name': 'None',
        'memory_mb': 0,
        'driver_version': 'N/A',
        'cuda_version': 'N/A',
        'compute_capability': 'N/A',
        'sm_count': 0,
    }
    
    try:
        import cupy as cp
        if cp.cuda.is_available():
            info['available'] = True
            
            device = cp.cuda.Device()
            attrs = device.attributes
            
            # Memory
            free, total = cp.cuda.runtime.memGetInfo()
            info['memory_mb'] = int(total / (1024**2))
            
            # Compute capability
            major = attrs.get('ComputeCapabilityMajor', 0)
            minor = attrs.get('ComputeCapabilityMinor', 0)
            info['compute_capability'] = f"{major}.{minor}"
            
            # SM count
            info['sm_count'] = attrs.get('MultiProcessorCount', 0)
            
            # CUDA version
            cuda_ver = cp.cuda.runtime.runtimeGetVersion()
            info['cuda_version'] = f"{cuda_ver // 1000}.{(cuda_ver % 1000) // 10}"
            
            # GPU name via pynvml
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(handle)
                info['name'] = name.decode('utf-8') if isinstance(name, bytes) else name
                
                driver = pynvml.nvmlSystemGetDriverVersion()
                info['driver_version'] = driver.decode('utf-8') if isinstance(driver, bytes) else driver
            except Exception:
                info['name'] = 'Unknown GPU'
    except ImportError:
        pass
    
    return info


def get_system_info() -> Dict[str, Any]:
    """Get complete system information."""
    return {
        'timestamp': datetime.now().isoformat(),
        'platform': sys.platform,
        'python_version': sys.version.split()[0],
        'sunburst_version': SUNBURST_VERSION,
        'cpu': get_cpu_info(),
        'gpu': get_gpu_info(),
    }


# =============================================================================
# HEADER GENERATION
# =============================================================================

def generate_header(
    test_name: str,
    version: str,
    config: Dict[str, Any],
    extra_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate standardized header for benchmark output files.
    
    Parameters
    ----------
    test_name : str
        Name of the test suite (e.g., "GPU Profiler", "Benchmark Suite")
    version : str
        Version of the test suite
    config : dict
        Configuration dictionary
    extra_info : dict, optional
        Additional key-value pairs to include
    
    Returns
    -------
    str
        Formatted header string
    """
    sys_info = get_system_info()
    cpu = sys_info['cpu']
    gpu = sys_info['gpu']
    
    # Build header
    lines = [
        "=" * 70,
        f"SUNBURST {test_name.upper()} v{version} SESSION LOG",
        "=" * 70,
        "",
        f"Timestamp: {sys_info['timestamp']}",
        f"SunBURST Package: {sys_info['sunburst_version']}",
        "",
        "CONFIGURATION:",
    ]
    
    for key, value in config.items():
        if isinstance(value, list):
            value_str = ', '.join(map(str, value))
            lines.append(f"  {key}: [{value_str}]")
        else:
            lines.append(f"  {key}: {value}")
    
    # CPU info
    lines.extend([
        "",
        "CPU INFORMATION:",
        f"  Name: {cpu['cpu_name']}",
        f"  Architecture: {cpu['architecture']}",
        f"  Physical Cores: {cpu['physical_cores'] or 'Unknown'}",
        f"  Logical Cores: {cpu['logical_cores']}",
    ])
    
    # GPU info
    lines.extend([
        "",
        "GPU INFORMATION:",
        f"  Available: {gpu['available']}",
        f"  Name: {gpu['name']}",
        f"  Memory: {gpu['memory_mb']} MB",
        f"  CUDA Version: {gpu['cuda_version']}",
        f"  Compute Capability: {gpu['compute_capability']}",
        f"  SM Count: {gpu['sm_count']}",
    ])
    
    # Extra info
    if extra_info:
        lines.extend(["", "ADDITIONAL INFO:"])
        for key, value in extra_info.items():
            lines.append(f"  {key}: {value}")
    
    lines.extend([
        "",
        "=" * 70,
        ""
    ])
    
    return '\n'.join(lines)


def generate_csv_header(
    test_name: str,
    version: str,
    config: Dict[str, Any]
) -> str:
    """
    Generate a compact header comment for CSV files.
    
    Returns a string like:
    # SunBURST GPU Profiler v3.0 | sunburst=1.0.0 | n_osc=1 | 2026-01-26T12:34:56
    """
    sys_info = get_system_info()
    
    parts = [
        f"# SunBURST {test_name} v{version}",
        f"sunburst={sys_info['sunburst_version']}",
        f"n_osc={config.get('n_oscillations', '?')}",
        sys_info['timestamp'][:19],
    ]
    
    return ' | '.join(parts)


# =============================================================================
# MAIN (self-test)
# =============================================================================

if __name__ == "__main__":
    print("Testing header_utils...\n")
    
    # Test system info
    sys_info = get_system_info()
    print("System Info:")
    print(f"  Platform: {sys_info['platform']}")
    print(f"  Python: {sys_info['python_version']}")
    print(f"  SunBURST: {sys_info['sunburst_version']}")
    print(f"  CPU: {sys_info['cpu']['cpu_name']}")
    print(f"  GPU: {sys_info['gpu']['name']}")
    
    # Test header generation
    header = generate_header(
        test_name="GPU Profiler",
        version="3.0",
        config={
            'n_oscillations': 1,
            'fast': True,
            'runs_per_dimension': 4,
            'dimensions': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        }
    )
    
    print("\nGenerated Header:")
    print(header)
