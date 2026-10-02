#!/usr/bin/env python3
"""
================================================================================
REPRODUCIBILITY MODULE - SEED MANAGEMENT
================================================================================
Master's Thesis - Electrical Engineering, Wuhan University
Ensures reproducible results across all experiments.

Usage:
------
    from src.utils.reproducibility import set_global_seed, get_seed_config
    
    # Set all seeds at the start of your script
    set_global_seed(42)
    
    # Or use the config
    config = get_seed_config()
    set_global_seed(config['global_seed'])

================================================================================
"""

import os
import random
import numpy as np

# Check for optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# =============================================================================
# DEFAULT SEED CONFIGURATION
# =============================================================================

SEED_CONFIG = {
    # Master seed - used for all random operations
    'global_seed': 42,
    
    # Component-specific seeds (derived from global for traceability)
    'numpy_seed': 42,
    'random_seed': 42,
    'torch_seed': 42,
    'tf_seed': 42,
    
    # Data splitting seeds
    'train_val_split_seed': 42,
    'test_split_seed': 42,
    'cv_fold_seed': 42,
    
    # Model-specific seeds
    'patchtst_seed': 42,
    'nhits_seed': 42,
    'lstm_seed': 42,
    'gru_seed': 42,
    'lightgbm_seed': 42,
    'xgboost_seed': 42,
    'ahse_seed': 42,
    
    # Data augmentation / noise seeds
    'augmentation_seed': 42,
    'noise_seed': 42,
    
    # Visualization seeds (for consistent synthetic examples)
    'viz_seed': 42,
}


def get_seed_config():
    """Return the seed configuration dictionary."""
    return SEED_CONFIG.copy()


def set_global_seed(seed: int = 42, deterministic: bool = True):
    """
    Set seeds for all random number generators for reproducibility.
    
    Parameters:
    -----------
    seed : int
        The seed value to use (default: 42)
    deterministic : bool
        If True, enables deterministic mode in PyTorch (may impact performance)
    
    Returns:
    --------
    dict : Configuration of seeds that were set
    
    Example:
    --------
    >>> from src.utils.reproducibility import set_global_seed
    >>> set_global_seed(42)
    {'python_random': 42, 'numpy': 42, 'torch': 42, 'cuda': 42}
    """
    seeds_set = {}
    
    # Python's built-in random
    random.seed(seed)
    seeds_set['python_random'] = seed
    
    # NumPy
    np.random.seed(seed)
    seeds_set['numpy'] = seed
    
    # Environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    seeds_set['python_hash'] = seed
    
    # PyTorch
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        seeds_set['torch'] = seed
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
            seeds_set['cuda'] = seed
            
            if deterministic:
                # Enable deterministic algorithms
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                seeds_set['cudnn_deterministic'] = True
                
                # PyTorch 1.8+ deterministic flag
                if hasattr(torch, 'use_deterministic_algorithms'):
                    try:
                        # Required for CUDA >= 10.2 deterministic behavior
                        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
                        torch.use_deterministic_algorithms(True)
                        seeds_set['torch_deterministic'] = True
                    except RuntimeError:
                        # Some operations don't have deterministic implementations
                        pass
    
    # TensorFlow
    if TF_AVAILABLE:
        tf.random.set_seed(seed)
        seeds_set['tensorflow'] = seed
    
    print(f"🎲 Global seed set to {seed}")
    print(f"   Seeds configured: {list(seeds_set.keys())}")
    
    return seeds_set


def get_worker_init_fn(seed: int = 42):
    """
    Get a worker initialization function for PyTorch DataLoader.
    
    Ensures reproducibility when using multiple workers.
    
    Usage:
    ------
    >>> from src.utils.reproducibility import get_worker_init_fn
    >>> loader = DataLoader(dataset, num_workers=4, worker_init_fn=get_worker_init_fn(42))
    """
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    return worker_init_fn


def get_generator(seed: int = 42):
    """
    Get a PyTorch Generator for reproducible data loading.
    
    Usage:
    ------
    >>> from src.utils.reproducibility import get_generator
    >>> g = get_generator(42)
    >>> loader = DataLoader(dataset, generator=g, shuffle=True)
    """
    if TORCH_AVAILABLE:
        g = torch.Generator()
        g.manual_seed(seed)
        return g
    return None


class ReproducibleRandom:
    """
    Context manager for reproducible random operations.
    
    Temporarily sets seeds for a block of code, then restores the previous state.
    
    Usage:
    ------
    >>> with ReproducibleRandom(seed=123):
    ...     # Random operations here are reproducible
    ...     x = np.random.randn(100)
    >>> # Previous random state is restored
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.numpy_state = None
        self.random_state = None
        self.torch_state = None
        self.cuda_state = None
    
    def __enter__(self):
        # Save current states
        self.numpy_state = np.random.get_state()
        self.random_state = random.getstate()
        
        if TORCH_AVAILABLE:
            self.torch_state = torch.get_rng_state()
            if torch.cuda.is_available():
                self.cuda_state = torch.cuda.get_rng_state_all()
        
        # Set seeds
        set_global_seed(self.seed, deterministic=False)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous states
        np.random.set_state(self.numpy_state)
        random.setstate(self.random_state)
        
        if TORCH_AVAILABLE:
            torch.set_rng_state(self.torch_state)
            if torch.cuda.is_available() and self.cuda_state is not None:
                torch.cuda.set_rng_state_all(self.cuda_state)
        
        return False


# =============================================================================
# MODEL-SPECIFIC SEED HELPERS
# =============================================================================

def get_lightgbm_params(seed: int = None):
    """Get LightGBM parameters with reproducibility settings."""
    if seed is None:
        seed = SEED_CONFIG['lightgbm_seed']
    
    return {
        'seed': seed,
        'bagging_seed': seed,
        'feature_fraction_seed': seed,
        'data_random_seed': seed,
        'deterministic': True,
        'force_row_wise': True,  # Required for deterministic mode
    }


def get_xgboost_params(seed: int = None):
    """Get XGBoost parameters with reproducibility settings."""
    if seed is None:
        seed = SEED_CONFIG['xgboost_seed']
    
    return {
        'seed': seed,
        'random_state': seed,
    }


def get_sklearn_params(seed: int = None):
    """Get scikit-learn parameters with reproducibility settings."""
    if seed is None:
        seed = SEED_CONFIG['global_seed']
    
    return {
        'random_state': seed,
    }


# =============================================================================
# INITIALIZATION
# =============================================================================

def init_reproducibility(seed: int = 42, verbose: bool = True):
    """
    Full initialization for reproducible experiments.
    
    Call this at the very beginning of your script.
    
    Parameters:
    -----------
    seed : int
        Global seed value
    verbose : bool
        Print configuration details
    
    Example:
    --------
    >>> from src.utils.reproducibility import init_reproducibility
    >>> init_reproducibility(42)
    """
    # Update global config
    SEED_CONFIG['global_seed'] = seed
    for key in SEED_CONFIG:
        if key != 'global_seed':
            SEED_CONFIG[key] = seed
    
    # Set all seeds
    seeds_set = set_global_seed(seed, deterministic=True)
    
    if verbose:
        print("\n" + "="*60)
        print("REPRODUCIBILITY CONFIGURATION")
        print("="*60)
        print(f"Global Seed: {seed}")
        print(f"NumPy: {np.__version__}")
        if TORCH_AVAILABLE:
            print(f"PyTorch: {torch.__version__}")
            print(f"CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print("="*60 + "\n")
    
    return seeds_set


if __name__ == "__main__":
    # Test reproducibility
    init_reproducibility(42)
    
    print("Testing reproducibility...")
    
    # Test 1: NumPy
    np.random.seed(42)
    a1 = np.random.randn(5)
    np.random.seed(42)
    a2 = np.random.randn(5)
    assert np.allclose(a1, a2), "NumPy reproducibility failed!"
    print("✓ NumPy reproducibility OK")
    
    # Test 2: Python random
    random.seed(42)
    b1 = [random.random() for _ in range(5)]
    random.seed(42)
    b2 = [random.random() for _ in range(5)]
    assert b1 == b2, "Python random reproducibility failed!"
    print("✓ Python random reproducibility OK")
    
    # Test 3: PyTorch (if available)
    if TORCH_AVAILABLE:
        torch.manual_seed(42)
        c1 = torch.randn(5)
        torch.manual_seed(42)
        c2 = torch.randn(5)
        assert torch.allclose(c1, c2), "PyTorch reproducibility failed!"
        print("✓ PyTorch reproducibility OK")
    
    print("\n✅ All reproducibility tests passed!")
