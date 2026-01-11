"""
Geometric Algebra for Representation, Robustness, and Explainability in Medical Image Classification

A framework for investigating geometric algebra-based representations in medical imaging benchmarks,
focusing on representation expressiveness, data efficiency, robustness, and intrinsic explainability.
Medical imaging serves as the testbed for evaluating algorithmic properties, not clinical utility.
"""

__version__ = "0.1.0"

# Core classes
from .ga_representation import (
    GeometricAlgebraRepresentation,
    GAMultivectorLayer,
    GAFeatureExtractor
)

from .model import (
    GAMedicalClassifier,
    GAMedicalClassifierWithAttention
)

from .explainability import GAExplainabilityAnalyzer

from .data_utils import (
    MedicalImageDataset,
    load_dataset_from_directory,
    create_dummy_dataset
)

from .device_utils import (
    get_device,
    print_device_info
)

from .preprocessing import (
    get_train_transform,
    get_val_transform,
    get_test_transform,
    STANDARD_PREPROCESSING
)

from .gradcam import (
    GradCAM,
    find_target_layer,
    create_gradcam_for_cnn
)

from .explainability_comparison import ExplainabilityComparator

from .statistical_comparison import (
    compare_models_statistically,
    compare_multiple_metrics,
    generate_statistical_report
)

from .reproducibility import (
    set_seed,
    set_deterministic
)

from .imbalance_handling import (
    calculate_class_weights,
    create_weighted_sampler,
    create_weighted_loss,
    create_focal_loss,
    get_imbalance_handling_strategy,
    print_imbalance_info,
    FocalLoss
)

# Make key classes available at package level
__all__ = [
    # Version
    '__version__',
    
    # GA Representation
    'GeometricAlgebraRepresentation',
    'GAMultivectorLayer',
    'GAFeatureExtractor',
    
    # Models
    'GAMedicalClassifier',
    'GAMedicalClassifierWithAttention',
    
    # Explainability
    'GAExplainabilityAnalyzer',
    
    # Data utilities
    'MedicalImageDataset',
    'load_dataset_from_directory',
    'create_dummy_dataset',
    
    # Device utilities
    'get_device',
    'print_device_info',
    
    # Preprocessing
    'get_train_transform',
    'get_val_transform',
    'get_test_transform',
    'STANDARD_PREPROCESSING',
    
    # Grad-CAM
    'GradCAM',
    'find_target_layer',
    'create_gradcam_for_cnn',
    
    # Explainability comparison
    'ExplainabilityComparator',
    
    # Statistical comparison
    'compare_models_statistically',
    'compare_multiple_metrics',
    'generate_statistical_report',
    
    # Reproducibility
    'set_seed',
    'set_deterministic',
    
    # Imbalance handling
    'calculate_class_weights',
    'create_weighted_sampler',
    'create_weighted_loss',
    'create_focal_loss',
    'get_imbalance_handling_strategy',
    'print_imbalance_info',
    'FocalLoss',
]

