# Geometric Algebra for Explainable AI in Medical Imaging

A comprehensive implementation of **Geometric Algebra (GA)-based Explainable AI** for medical image classification, focusing on **intrinsic explainability** and **fair comparison** with traditional CNN approaches.

## 🎯 Project Overview

This work investigates whether geometric algebra–based representations can offer more transparent and robust learning behavior than conventional deep learning approaches in medical imaging benchmarks. The project evaluates:

- **Representation expressiveness**: GA multivectors vs pixel tensors
- **Intrinsic explainability**: GA geometric components vs post-hoc methods (Grad-CAM)
- **Fair comparison**: Same dataset, same evaluation protocol, same metrics
- **Statistical validation**: Significance testing and effect sizes

**Note**: This work does not aim to replace clinical diagnostic systems, but rather investigates algorithmic properties of geometric algebra representations in medical imaging benchmarks.

### Key Innovation

Unlike traditional CNNs that treat images as flat pixel arrays, our approach:
- Encodes **explicit geometric structure** (scalars, vectors, bivectors, trivectors)
- Provides **intrinsic explainability** through algebraic component decomposition
- Maintains **interpretability** throughout the learning pipeline
- Enables **geometric reasoning** about image features in a structured representation space

## 📚 Geometric Algebra Concepts

The system uses Clifford Algebra to represent medical images as **multivectors** in GA(3) space:

- **Scalars (Grade 0)**: Normalized pixel intensity
- **Vectors (Grade 1)**: Spatial gradients (dx, dy)
- **Bivectors (Grade 2)**: Orientations and textures (second-order derivatives)
- **Trivector (Grade 3)**: Complex geometric relationships

This representation captures rich geometric features that are naturally interpretable.

## 🏗️ Project Structure

```
MastersGA/
├── ga_medical_imaging/          # Main package
│   ├── ga_representation.py    # Image → multivector conversion
│   ├── model.py                 # GA-based classification models
│   ├── explainability.py        # Intrinsic explainability module
│   ├── gradcam.py               # Grad-CAM for CNN baselines
│   ├── explainability_comparison.py  # GA vs Grad-CAM comparison
│   ├── statistical_comparison.py    # Significance testing
│   ├── preprocessing.py         # Standardized preprocessing
│   ├── imbalance_handling.py   # Class imbalance solutions
│   ├── data_utils.py            # Dataset loading
│   ├── evaluate_5fold_cv.py    # 5-fold cross-validation
│   ├── compare_models.py        # GA vs CNN comparison
│   ├── train_professional.py   # Professional training script
│   └── reproducibility.py      # Reproducibility utilities
├── GA_Medical_Imaging_Colab.ipynb  # Interactive Colab notebook
├── experiments/                 # Experimental scripts
├── data/                       # Dataset directory
└── README.md                   # This file
```

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. Open **[GA_Medical_Imaging_Colab.ipynb](GA_Medical_Imaging_Colab.ipynb)** on [Google Colab](https://colab.research.google.com/)
2. Run cells in order
3. The notebook includes complete workflow:
   - Baseline comparison
   - GA framework demonstration
   - Training and evaluation
   - Explainability comparison (GA vs Grad-CAM)
   - Statistical analysis

### Option 2: Local Installation

#### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

#### Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .  # Install package in development mode
```

## 💻 Usage

### 1. Dataset Preparation

#### Using COVID-19 Chest X-Ray Dataset

```bash
# Download and organize the dataset
python -m ga_medical_imaging.organize_covid_dataset \
    --dataset_dir data/covid-chestxray-dataset \
    --output_dir data/covid_chestxray
```

**Dataset Structure:**
```
data/covid_chestxray/
├── covid/              # COVID-19 positive cases (label=1)
└── no_findings/        # Normal/negative cases (label=0)
```

**Note**: The dataset has class imbalance (1150 COVID vs 44 No Finding, ratio 26:1). See [Class Imbalance Handling](#class-imbalance-handling) section.

### 2. Training the Model

#### Professional Training (Recommended)

```bash
python -m ga_medical_imaging.train_professional \
    --data_dir data/covid_chestxray \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --save_dir checkpoints
```

Features:
- Early stopping
- Learning rate scheduling (Cosine Annealing with Warm Restarts)
- Mixed precision training (CUDA)
- Gradient clipping
- Class-weighted loss (handles imbalance automatically)

### 3. Evaluation

#### 5-Fold Cross-Validation (Matching Baseline Protocol)

```bash
python -m ga_medical_imaging.evaluate_5fold_cv \
    --data_dir data/covid_chestxray \
    --num_epochs 30 \
    --output_dir results/5fold_cv
```

This generates:
- Accuracy, Sensitivity, Specificity, Precision, F1-Score, ROC AUC
- Mean ± std across 5 folds
- Results saved in JSON format
- **Automatically uses class-weighted loss** to handle imbalance

### 4. Explainability Comparison

#### Compare GA vs Grad-CAM

```python
from ga_medical_imaging.compare_models import TraditionalCNN
from ga_medical_imaging.gradcam import create_gradcam_for_cnn
from ga_medical_imaging.explainability_comparison import ExplainabilityComparator

# Train or load both models
ga_model = ...  # Your trained GA model
cnn_model = TraditionalCNN(num_classes=2, input_channels=1).to(device)
# ... train CNN model ...

# Create comparator
gradcam = create_gradcam_for_cnn(cnn_model)
comparator = ExplainabilityComparator(
    ga_model=ga_model,
    cnn_model=cnn_model,
    gradcam=gradcam,
    device=device
)

# Compare explanations
comparison = comparator.compare_explanations(image, target_class=label)
comparator.visualize_comparison(image, comparison, save_path='comparison.png')
```

### 5. Statistical Significance Testing

```python
from ga_medical_imaging.statistical_comparison import (
    compare_multiple_metrics,
    generate_statistical_report
)

# After running 5-fold CV for both models
ga_results = ...  # From evaluate_5fold_cv
cnn_results = ...  # From evaluate_5fold_cv

# Compare all metrics
comparison = compare_multiple_metrics(ga_results, cnn_results)

# Generate report
report = generate_statistical_report(
    'GA Model',
    'CNN Baseline',
    comparison,
    output_path='results/statistical_comparison.txt'
)
```

## ⚖️ Fair Comparison Methodology

### What MUST Match Exactly

1. **Dataset**: Use `ieee8023/covid-chestxray-dataset` (same as DarkCovidNet, VGG-19)
2. **Evaluation Protocol**: 5-fold cross-validation (matching DarkCovidNet)
3. **Metrics**: Accuracy, Sensitivity, Specificity, Precision, F1-Score, ROC AUC
4. **Preprocessing**: Standardized transforms (see `preprocessing.py`)
5. **Task**: Binary classification (COVID-19 vs no-findings)

### What CAN Differ (But Should Be Fair)

1. **Model Architecture**: Different architectures are expected (GA vs CNN)
2. **Training Procedure**: Can differ, but should be:
   - Similar computational budget
   - Both optimized (not one optimized, one default)
   - Both use best practices (early stopping, LR scheduling)

### Recommended Reporting Structure

Report results in two categories:

**1. Primary Comparison (No Augmentation)**
- GA model vs DarkCovidNet, VGG-19
- Same original dataset
- Same evaluation protocol

**2. Explainability Comparison**
- GA intrinsic explanations vs Grad-CAM (post-hoc)
- Quantitative metrics: spatial correlation, overlap, sparsity
- Statistical significance testing

### Ensuring Fairness

✅ **Already implemented**:
- Standardized preprocessing (`preprocessing.py`)
- Same evaluation protocol (5-fold CV)
- Class-weighted loss for both models (fair imbalance handling)
- Reproducibility utilities (`reproducibility.py`)

## ⚖️ Class Imbalance Handling

### The Problem

Your dataset has severe imbalance:
- COVID-19: 1150 images (96.3%)
- No Finding: 44 images (3.7%)
- **Imbalance ratio: 26:1** ⚠️

### ✅ Recommended Solution: Class-Weighted Loss

**Why it's better than cGAN**:
- ✅ Stable and reliable (no collapse)
- ✅ Simple (one line of code)
- ✅ No extra training time
- ✅ Fair comparison (both models use same weights)

**How it works**:
- Automatically penalizes misclassifying the minority class more heavily
- Weights calculated as: `weight[i] = total_samples / (num_classes * class_i_count)`

**Usage**:
```python
from ga_medical_imaging.imbalance_handling import create_weighted_loss, print_imbalance_info

# Analyze imbalance
labels = [dataset[i][1] for i in range(len(dataset))]
print_imbalance_info(labels)

# Create weighted loss
criterion = create_weighted_loss(labels, device=device)

# Use in training (automatically handles imbalance)
```

**Already integrated**: `evaluate_5fold_cv.py` uses weighted loss by default!

### Alternative Solutions

1. **Weighted Random Sampling**: Balances batches during training
2. **Focal Loss**: Focuses on hard examples
3. **Combined Approach**: Both weighted sampling + weighted loss

See `imbalance_handling.py` for all options.

## 📊 Evaluation Framework

### Core Metrics

- **Accuracy**: Overall classification correctness
- **Sensitivity (Recall)**: True Positive Rate
- **Specificity**: True Negative Rate
- **Precision**: Positive Predictive Value
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under ROC curve

### Evaluation Protocol

For fair comparison:
- **5-fold Cross-Validation**: Matching DarkCovidNet protocol
- **Standardized Preprocessing**: Same transforms for all models
- **Class-Weighted Loss**: Handles imbalance fairly
- **Statistical Testing**: Validate significance of differences

### Baseline Methods

We compare against:
- **DarkCovidNet**: Accuracy ≈ 98%, Sensitivity ≈ 95%, Specificity ≈ 95%, Precision ≈ 98%, F1 ≈ 0.97
- **VGG-19**: Accuracy ≈ 98.75%
- **VGG16 + cGAN**: Accuracy ≈ 99.76% (Electronics 2022) - Reference only

## 🔬 Model Architecture

### GAMedicalClassifier

The main model consists of:

1. **GeometricAlgebraRepresentation**: Converts images → multivectors (8 components)
2. **GAFeatureExtractor**: Extracts geometric features via GA layers
   - `GAMultivectorLayer`: Neural layers operating on multivectors
   - Geometric product adapted for PyTorch
3. **Classifier**: Final classification layers with BatchNorm and Dropout

### Architecture Flow

```
Input Image (B, C, H, W)
    ↓
GeometricAlgebraRepresentation
    ↓
Multivectors (B, H, W, 8)
    ↓
GAFeatureExtractor
    ├─ GAMultivectorLayer(1 → 64)
    ├─ ReLU
    ├─ GAMultivectorLayer(64 → 128)
    ├─ ReLU
    ├─ GAMultivectorLayer(128 → 256)
    └─ Projection (with BatchNorm, Dropout)
    ↓
Features (B, 256)
    ↓
Classifier
    ├─ Linear(256 → 128) + BatchNorm + ReLU + Dropout
    ├─ Linear(128 → 64) + BatchNorm + ReLU + Dropout
    └─ Linear(64 → num_classes)
    ↓
Logits (B, num_classes)
```

## 📈 Intrinsic Explainability

### Key Advantage

Unlike post-hoc methods (Grad-CAM, SHAP, LIME), our approach provides **intrinsic explainability**:

- **Direct Access**: Explanations based on actual model structure
- **Algebraic Decomposition**: Decisions explained in terms of geometric components
- **Stability**: Explanations are stable under small input perturbations
- **No Approximation**: Direct access to representation components
- **Structured**: Explanations decompose into interpretable geometric components

### Explainability Features

1. **Geometric Component Analysis**
   - Contribution of scalars (pixel intensities)
   - Contribution of vectors (spatial gradients)
   - Contribution of bivectors (orientations and textures)
   - Contribution of trivector (complex relationships)

2. **Spatial Importance Maps**
   - Multivector magnitude visualization
   - Salient feature identification

3. **Comparison with Grad-CAM**
   - Quantitative metrics: spatial correlation, overlap, sparsity
   - Side-by-side visualization
   - Stability analysis

### Example Explanation Output

```
=== EXPLANATION REPORT ===

PREDICTION:
  Predicted class: 1 (COVID-19)
  Confidence: 87.3%
  
GEOMETRIC COMPONENT CONTRIBUTIONS:

1. Scalars (Pixel intensities): 25.3%
2. Vectors (Spatial gradients): 30.1%
3. Bivectors (Orientations and textures): 35.2%
4. Trivector (Complex relationships): 9.4%

ANALYSIS:
The most influential component is orientations and textures 
(35.2% of total contribution), indicating that texture features
are the primary contributing factors in this prediction.
```

## 📁 File Structure & Workflow

### Core Modules

**Foundation**:
- `reproducibility.py` - Set random seeds for reproducible experiments
- `device_utils.py` - Device detection (CPU/CUDA/MPS)
- `preprocessing.py` - Standardized preprocessing transforms

**Data**:
- `data_utils.py` - Dataset loading and management
- `imbalance_handling.py` - Class imbalance solutions
- `organize_covid_dataset.py` - Dataset organization utility

**GA Core**:
- `ga_representation.py` - Image → multivector conversion, GA layers
- `model.py` - GA-based classification models

**Explainability**:
- `explainability.py` - GA intrinsic explainability
- `gradcam.py` - Grad-CAM for CNN baselines
- `explainability_comparison.py` - GA vs Grad-CAM comparison

**Evaluation**:
- `evaluate_5fold_cv.py` - 5-fold cross-validation
- `statistical_comparison.py` - Significance testing
- `compare_models.py` - GA vs CNN direct comparison
- `compare_with_baselines.py` - Compare with published results

**Training**:
- `train_professional.py` - Professional training with best practices

### Typical Workflow

```
1. Setup & Data Preparation
   reproducibility.py → Set seeds
   preprocessing.py → Define transforms
   data_utils.py → Load dataset
   imbalance_handling.py → Handle class imbalance

2. Model Definition
   ga_representation.py → Define GA representation
   model.py → Create GA model

3. Training
   train_professional.py → Train model

4. Evaluation
   evaluate_5fold_cv.py → Run 5-fold CV
   statistical_comparison.py → Test significance

5. Explainability Comparison
   explainability.py → Analyze GA explanations
   gradcam.py → Generate Grad-CAM for CNN
   explainability_comparison.py → Compare both

6. Reporting
   compare_models.py → Direct GA vs CNN comparison
   compare_with_baselines.py → Compare with literature
```

## 🔧 Configuration

### Training Parameters

- `--num_epochs`: Number of epochs (default: 50)
- `--batch_size`: Batch size (default: 16)
- `--learning_rate`: Learning rate (default: 0.001)
- `--image_size`: Image size (default: 224 224)
- `--early_stopping_patience`: Early stopping patience (default: 15)
- `--weight_decay`: L2 regularization (default: 1e-4)

### Model Parameters

- `multivector_dim`: Multivector dimension (8 for GA 3D)
- `feature_dim`: Extracted feature dimension (256)
- `num_classes`: Number of classes (2 for binary)

### Evaluation Parameters

- `--n_splits`: Number of CV folds (default: 5)
- `--handle_imbalance`: Use class-weighted loss (default: True)

## 🧪 Testing

### Quick Test with Dummy Data

```python
from ga_medical_imaging.data_utils import create_dummy_dataset
from ga_medical_imaging.model import GAMedicalClassifier

# Create dummy dataset
image_paths, labels = create_dummy_dataset(num_samples=100)

# Create and test model
model = GAMedicalClassifier(num_classes=2, device='cpu')
# ... training and evaluation
```

## 🔬 Research Contributions

### Main Contributions

1. **Multivector Representation for Image Classification**
   - Explicit geometric structure encoding
   - Scalars, vectors, bivectors, trivectors
   - Code: `ga_representation.py::GeometricAlgebraRepresentation`

2. **Neural Layers on Multivectors**
   - Specialized `GAMultivectorLayer` operating on multivectors
   - Geometric product adapted for PyTorch
   - Code: `ga_representation.py::GAMultivectorLayer`

3. **Intrinsic Explainability**
   - Explanations based on model structure
   - Direct algebraic component decomposition
   - Code: `explainability.py::GAExplainabilityAnalyzer`

4. **Fair Comparison Framework**
   - Standardized preprocessing
   - Same evaluation protocol
   - Statistical significance testing
   - Code: `evaluate_5fold_cv.py`, `statistical_comparison.py`

5. **Explainability Comparison**
   - GA intrinsic vs Grad-CAM (post-hoc)
   - Quantitative comparison metrics
   - Code: `explainability_comparison.py`

## 🎯 Benchmark Datasets

### Primary Dataset: COVID Chest X-ray

- **Source**: [ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset)
- **Task**: Binary classification (COVID-19 vs no-findings)
- **Statistics**: ~1150 COVID-19 cases, ~44 No Finding cases
- **Imbalance**: 26:1 ratio (handled with class-weighted loss)

## 📈 Performance Evaluation

### Evaluation Axes

1. **Representation Expressiveness**: GA vs pixel tensors
2. **Explainability**: Intrinsic vs post-hoc methods (Grad-CAM)
3. **Fair Comparison**: Same dataset, same protocol, same metrics
4. **Statistical Validation**: Significance testing and effect sizes

### Running Evaluations

**Standard classification metrics**:
```bash
python -m ga_medical_imaging.evaluate_5fold_cv --data_dir data/covid_chestxray
```

**Compare with CNN baselines**:
```bash
python -m ga_medical_imaging.compare_models --data_dir data/covid_chestxray
```

**Statistical significance testing**:
```python
from ga_medical_imaging.statistical_comparison import compare_multiple_metrics
comparison = compare_multiple_metrics(ga_results, cnn_results)
```

## 🔮 Future Improvements

- [ ] Support for 3D image volumes (GA(4) or higher)
- [ ] Advanced GA operations (full geometric product, rotors, versors)
- [ ] Hybrid architectures (GA layers + standard CNNs)
- [ ] Additional benchmark datasets
- [ ] Multi-class classification (beyond binary)
- [ ] Transfer learning experiments

## 📄 License

This project is intended for research and educational purposes.

**Dataset License**: Each image has its own license specified in metadata.csv (Apache 2.0, CC BY-NC-SA 4.0, CC BY 4.0)

## 📧 Citation

If you use this work, please cite:

```bibtex
@article{cohen2020covid,
  title={COVID-19 image data collection},
  author={Joseph Paul Cohen and Paul Morrison and Lan Dao},
  journal={arXiv 2003.11597},
  url={https://github.com/ieee8023/covid-chestxray-dataset},
  year={2020}
}
```

## 📖 References

### Geometric Algebra
- Hestenes, D. (1986). *New Foundations for Classical Mechanics*
- Dorst, L., Fontijne, D., & Mann, S. (2007). *Geometric Algebra for Computer Science*

### Explainable AI
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks
- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions

### Baseline Methods
- DarkCovidNet (MDPI Review)
- VGG-19 for COVID-19 classification
- VGG16 + cGAN (Electronics 2022): "Machine-Learning-Based COVID-19 Detection with Enhanced cGAN Technique Using X-ray Images"

### Dataset
- Cohen, J. P., et al. (2020). COVID-19 image data collection. arXiv:2003.11597

---

**Note**: This work investigates algorithmic properties of geometric algebra representations in medical imaging benchmarks. Medical imaging serves as the testbed, not the contribution. The focus is on representation learning, robustness, and explainability in CS/ML terms, not clinical diagnostic utility.

**Project Status**: Active development - See issues and pull requests for latest updates.
