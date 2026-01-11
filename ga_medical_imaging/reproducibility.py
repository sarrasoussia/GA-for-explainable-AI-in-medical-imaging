"""
Reproducibility utilities for consistent experimental results.

This module provides functions to set random seeds across all libraries
to ensure reproducible experiments.
"""

import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42):
    """
    Set all random seeds for reproducibility.
    
    Sets seeds for:
    - Python random
    - NumPy
    - PyTorch (CPU and CUDA)
    - CuDNN (if available)
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_deterministic():
    """
    Set PyTorch to deterministic mode for full reproducibility.
    
    Note: This may slow down training but ensures reproducibility.
    """
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

