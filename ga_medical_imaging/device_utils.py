"""
Device utility functions for automatic device selection.

Supports:
- MPS (Metal Performance Shaders) for Apple Silicon Macs
- CUDA for NVIDIA GPUs
- CPU as fallback
"""

import torch


def get_device(device: str = 'auto') -> str:
    """
    Get the best available device for PyTorch operations.
    
    Priority order:
    1. MPS (Metal) - for Apple Silicon Macs
    2. CUDA - for NVIDIA GPUs
    3. CPU - fallback
    
    Args:
        device: Device preference ('auto', 'mps', 'cuda', 'cpu')
        
    Returns:
        Device string ('mps', 'cuda', or 'cpu')
    """
    if device != 'auto':
        # User specified device explicitly
        if device == 'mps' and not torch.backends.mps.is_available():
            print(f"⚠ Warning: MPS requested but not available, falling back to CPU")
            return 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            print(f"⚠ Warning: CUDA requested but not available, falling back to CPU")
            return 'cpu'
        return device
    
    # Auto-detect best device
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def print_device_info(device: str):
    """
    Print information about the selected device.
    
    Args:
        device: Device string
    """
    print(f"\n{'='*70}")
    print(f"DEVICE INFORMATION")
    print(f"{'='*70}")
    print(f"Selected device: {device}")
    
    if device == 'mps':
        print(f"  Type: Apple Silicon GPU (Metal Performance Shaders)")
        print(f"  Status: Available and ready")
    elif device == 'cuda':
        print(f"  Type: NVIDIA GPU")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print(f"  Type: CPU")
        print(f"  Note: Training will be slower. Consider using GPU if available.")
    
    print(f"{'='*70}\n")

