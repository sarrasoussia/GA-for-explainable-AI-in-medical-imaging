# Geometric Algebra for Explainable AI in Medical Imaging

A comprehensive implementation of **Geometric Algebra (GA)-based Explainable AI** for medical image classification, specifically applied to **COVID-19 detection from chest X-ray images**.

## 🎯 Project Overview

This project introduces a novel approach that:
- **Represents medical images as multivectors** in Geometric Algebra space
- **Provides intrinsic explainability** through geometric component analysis
- **Achieves competitive performance** compared to state-of-the-art baseline methods
- **Offers interpretable insights** for clinical decision-making

### Key Innovation

Unlike traditional CNNs that treat images as flat pixel arrays, our approach:
- Encodes **explicit geometric structure** (scalars, vectors, bivectors, trivectors)
- Provides **intrinsic explainability** (not post-hoc approximations)
- Maintains **interpretability** throughout the learning pipeline
- Enables **geometric reasoning** about medical image features

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
├── ga_medical_imaging/
│   ├── __init__.py
│   ├── ga_representation.py          # Image → multivector conversion
│   ├── model.py                      # GA-based classification models
│   ├── explainability.py             # Intrinsic explainability module
│   ├── data_utils.py                 # Dataset loading and preprocessing
│   ├── train.py                      # Training script
│   ├── evaluate_and_explain.py      # Evaluation and explanation generation
│   ├── evaluate_5fold_cv.py          # 5-fold cross-validation evaluation
│   ├── compare_with_baselines.py    # Baseline comparison framework
│   ├── compare_models.py             # GA vs CNN comparison
│   ├── cgan_generator.py             # Conditional GAN for data augmentation
│   ├── train_cgan.py                 # cGAN training script
│   ├── generate_synthetic_dataset.py # Synthetic image generation
│   ├── organize_covid_dataset.py     # COVID-19 dataset organization
│   ├── run_baseline_comparison.py    # Complete evaluation pipeline
│   └── run_cgan_pipeline.py          # cGAN augmentation pipeline
├── GA_Medical_Imaging_Colab.ipynb    # Interactive Colab notebook
├── scripts/
│   └── generate_contribution_summary.py  # Auto-generate contribution summary
├── data/
│   └── covid_chestxray/              # Organized COVID-19 dataset
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🚀 Quick Start

### Option 1: Google Colab (Recommended for beginners)

The easiest way to explore the project:

1. Open **[GA_Medical_Imaging_Colab.ipynb](GA_Medical_Imaging_Colab.ipynb)** on [Google Colab](https://colab.research.google.com/)
2. Run cells in order
3. The notebook includes:
   - Baseline comparison with state-of-the-art methods
   - GA framework demonstration
   - Training and evaluation
   - Intrinsic explainability analysis

### Option 2: Local Installation

#### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

#### Install Dependencies

```bash
pip install -r requirements.txt
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
│   ├── image1.jpg
│   └── ...
└── no_findings/        # Normal/negative cases (label=0)
    ├── image1.jpg
    └── ...
```

**Note**: The dataset has class imbalance (575 COVID vs 22 No Finding). See [cGAN Augmentation](#cgan-based-data-augmentation) for balancing.

#### Using Your Own Data

Organize images in the following structure:
```
data/
├── covid/ or tumeur/ or positive/    # Positive class (label=1)
│   └── *.png, *.jpg
└── no_findings/ or sain/ or normal/   # Negative class (label=0)
    └── *.png, *.jpg
```

### 2. Training the Model

#### Basic Training

```bash
python -m ga_medical_imaging.train \
    --data_dir data/covid_chestxray \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 0.001
```

#### With Dummy Dataset (for testing)

```bash
python -m ga_medical_imaging.train --num_epochs 20
```

### 3. Evaluation

#### 5-Fold Cross-Validation (Matching Baseline Protocol)

```bash
python -m ga_medical_imaging.evaluate_5fold_cv \
    --data_dir data/covid_chestxray \
    --num_epochs 30 \
    --output_dir results/5fold_cv
```

This generates comprehensive metrics:
- Accuracy, Sensitivity, Specificity, Precision, F1-Score, ROC AUC
- Mean ± std across 5 folds
- Results saved in JSON format

#### Compare with Baselines

```bash
python -m ga_medical_imaging.compare_with_baselines \
    --cv_results results/5fold_cv/cv_results.json \
    --output_dir results/comparison
```

This compares your results with:
- **DarkCovidNet**: Accuracy ≈ 98%, Sensitivity ≈ 95%, Specificity ≈ 95%, Precision ≈ 98%, F1 ≈ 0.97
- **VGG-19**: Accuracy ≈ 98.75%
- **VGG16 + cGAN**: Accuracy ≈ 99.76% (Electronics 2022)

### 4. Explainability Analysis

```bash
python -m ga_medical_imaging.evaluate_and_explain \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/image.png \
    --output_dir explanations
```

Generates:
- Geometric component contribution analysis
- Spatial importance maps
- Visual explanations
- Textual explanation reports

## 🎨 cGAN-Based Data Augmentation

To handle class imbalance and match the Electronics 2022 approach:

### Complete Pipeline

```bash
python -m ga_medical_imaging.run_cgan_pipeline \
    --data_dir data/covid_chestxray
```

This automatically:
1. Trains a conditional GAN
2. Generates synthetic COVID-19 images
3. Creates a balanced dataset
4. Runs 5-fold CV evaluation
5. Compares with baselines

### Step-by-Step

```bash
# 1. Train cGAN
python -m ga_medical_imaging.train_cgan \
    --data_dir data/covid_chestxray \
    --num_epochs 100 \
    --save_dir checkpoints/cgan

# 2. Generate balanced dataset
python -m ga_medical_imaging.generate_synthetic_dataset \
    --original_data_dir data/covid_chestxray \
    --output_dir data/covid_chestxray_balanced \
    --cgan_checkpoint checkpoints/cgan/cgan_final.pth

# 3. Evaluate on balanced dataset
python -m ga_medical_imaging.evaluate_5fold_cv \
    --data_dir data/covid_chestxray_balanced
```

See **[CGAN_AUGMENTATION_GUIDE.md](guided%20reports/CGAN_AUGMENTATION_GUIDE.md)** for detailed documentation.

## 📊 Baseline Comparison

### Reported Baseline Results

| Method | Accuracy | Sensitivity | Specificity | Precision | F1-Score | Evaluation |
|--------|----------|-------------|-------------|-----------|----------|------------|
| **DarkCovidNet** | 98% | 95% | 95% | 98% | 0.97 | 5-fold CV |
| **VGG-19** | 98.75% | - | - | - | - | Not specified |
| **VGG16 + cGAN** | 99.76% | - | - | - | - | Not specified |

### Evaluation Protocol

For fair comparison, we use:
- **5-fold Cross-Validation**: Matching DarkCovidNet protocol
- **Same Dataset**: ieee8023/covid-chestxray-dataset
- **Comprehensive Metrics**: Accuracy, Sensitivity, Specificity, Precision, F1-Score, ROC AUC

## 🔬 Model Architecture

### GAMedicalClassifier

The main model consists of:

1. **GeometricAlgebraRepresentation**: Converts images → multivectors (8 components)
2. **GAFeatureExtractor**: Extracts geometric features via GA layers
   - `GAMultivectorLayer`: Neural layers operating on multivectors
   - Geometric product adapted for PyTorch
3. **Classifier**: Final classification layers

### Architecture Flow

```
Input Image (B, C, H, W)
    ↓
GeometricAlgebraRepresentation
    ↓
Multivectors (B, H, W, 8)
    ↓
GAFeatureExtractor
    ├─ GAMultivectorLayer(1 → 32)
    ├─ ReLU
    ├─ GAMultivectorLayer(32 → 64)
    ├─ ReLU
    ├─ GAMultivectorLayer(64 → 128)
    └─ Projection
    ↓
Features (B, 128)
    ↓
Classifier
    ├─ Linear(128 → 64)
    ├─ ReLU
    ├─ Dropout(0.3)
    └─ Linear(64 → num_classes)
    ↓
Logits (B, num_classes)
```

## 📈 Intrinsic Explainability

### Key Advantage

Unlike post-hoc methods (Grad-CAM, SHAP, LIME), our approach provides **intrinsic explainability**:

- **Direct Access**: Explanations based on actual model structure
- **Geometric Interpretability**: Decisions explained in terms of geometric concepts
- **Stability**: Explanations are stable under transformations
- **No Approximation**: Direct access to geometric components

### Explainability Features

1. **Geometric Component Analysis**
   - Contribution of scalars (intensities)
   - Contribution of vectors (gradients)
   - Contribution of bivectors (textures/orientations)
   - Contribution of trivector (complex relationships)

2. **Spatial Importance Maps**
   - Multivector magnitude visualization
   - Region importance identification

3. **Explanation Reports**
   - Quantitative component contributions
   - Qualitative interpretations
   - Clinical relevance analysis

### Example Explanation Output

```
=== EXPLANATION REPORT - MEDICAL DIAGNOSIS ===

PREDICTION:
  Predicted class: COVID-19
  Confidence: 87.3%
  
GEOMETRIC COMPONENT CONTRIBUTIONS:

1. Scalars (Pixel intensities):
   Contribution: 25.3%
   
2. Vectors (Spatial gradients):
   Contribution: 30.1%
   
3. Bivectors (Orientations and textures):
   Contribution: 35.2%
   
4. Trivector (Complex relationships):
   Contribution: 9.4%

ANALYSIS:
The most influential component is orientations and textures 
(35.2% of total contribution).
```

## 🔧 Configuration

### Training Parameters

- `--num_epochs`: Number of epochs (default: 50)
- `--batch_size`: Batch size (default: 16)
- `--learning_rate`: Learning rate (default: 0.001)
- `--image_size`: Image size (default: 224 224)

### Model Parameters

- `multivector_dim`: Multivector dimension (8 for GA 3D)
- `feature_dim`: Extracted feature dimension (128)
- `num_classes`: Number of classes (2 for binary)

### Evaluation Parameters

- `--n_splits`: Number of CV folds (default: 5)
- `--balance_ratio`: cGAN balance ratio (1.0 = fully balanced)

## 📊 Evaluation Metrics

The framework computes comprehensive metrics:

- **Accuracy**: Overall classification correctness
- **Sensitivity (Recall)**: True Positive Rate
- **Specificity**: True Negative Rate
- **Precision**: Positive Predictive Value
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under ROC curve

All metrics match baseline reporting format for fair comparison.

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

### Run Example Scripts

```bash
python example_usage.py
```

## 🔬 Research Contributions

This project presents several original contributions to explainable AI in medical imaging:

### Main Contributions

1. **Multivector Representation for Medical Images**
   - Explicit geometric structure encoding
   - Scalars, vectors, bivectors, trivectors
   - Code: `ga_representation.py::GeometricAlgebraRepresentation`

2. **Neural Layers on Multivectors**
   - Specialized `GAMultivectorLayer` operating on multivectors
   - Geometric product adapted for PyTorch
   - Code: `ga_representation.py::GAMultivectorLayer`

3. **Intrinsic Explainability**
   - Explanations based on model structure
   - Not post-hoc approximations
   - Code: `explainability.py::GAExplainabilityAnalyzer`

4. **Baseline Comparison Framework**
   - Comprehensive comparison with state-of-the-art
   - Same evaluation protocol
   - Code: `compare_with_baselines.py`, `evaluate_5fold_cv.py`

5. **cGAN-Based Augmentation**
   - Conditional GAN for class balancing
   - Matches Electronics 2022 approach
   - Code: `cgan_generator.py::ConditionalGAN`

6. **End-to-End Explainable Architecture**
   - Maintains interpretability at every stage
   - White-box design vs. traditional black-box
   - Code: `model.py::GAMedicalClassifier`

For detailed contributions, see:
- **[CONTRIBUTIONS.md](guided%20reports/CONTRIBUTIONS.md)**: Detailed contributions
- **[CONTRIBUTION_SUMMARY.md](CONTRIBUTION_SUMMARY.md)**: Auto-generated summary
- **[RESEARCH_REPORT.md](guided%20reports/RESEARCH_REPORT.md)**: Complete research report

## 📚 Documentation

### Research Documentation

- **[RESEARCH_REPORT.md](guided%20reports/RESEARCH_REPORT.md)**: Complete research report
- **[RESEARCH_PAPER_OUTLINE.md](guided%20reports/RESEARCH_PAPER_OUTLINE.md)**: Paper outline
- **[CONTRIBUTIONS.md](guided%20reports/CONTRIBUTIONS.md)**: Detailed contributions
- **[QUICKSTART.md](guided%20reports/QUICKSTART.md)**: Quick start guide

### Evaluation & Comparison

- **[CGAN_AUGMENTATION_GUIDE.md](guided%20reports/CGAN_AUGMENTATION_GUIDE.md)**: cGAN augmentation guide
- **Baseline Comparison**: Integrated in evaluation scripts

### Auto-Generated

Generate automatic contribution summary:
```bash
python scripts/generate_contribution_summary.py
```

## 🎯 Application: COVID-19 Detection

### Dataset

- **Source**: [ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset)
- **Task**: Binary classification (COVID-19 vs no-findings)
- **Statistics**: ~575 COVID-19 cases, ~22 no-findings cases
- **Class Imbalance**: Handled via cGAN augmentation

### Baseline Methods

1. **DarkCovidNet** (MDPI Review)
   - 5-fold CV, Accuracy ≈ 98%

2. **VGG-19**
   - Accuracy ≈ 98.75%

3. **VGG16 + cGAN** (Electronics 2022)
   - Accuracy ≈ 99.76%
   - Uses cGAN for class balancing

## 📈 Performance

### Expected Results

Based on baseline comparisons and evaluation protocol:
- **Accuracy**: Comparable to or exceeding baseline methods
- **Explainability**: Intrinsic geometric component analysis
- **Robustness**: Improved under geometric transformations

### Evaluation Results

Run 5-fold CV to get your results:
```bash
python -m ga_medical_imaging.evaluate_5fold_cv --data_dir data/covid_chestxray
```

Compare with baselines:
```bash
python -m ga_medical_imaging.compare_with_baselines \
    --cv_results results/5fold_cv/cv_results.json
```

## 🔮 Future Improvements

- [ ] Support for 3D medical volumes (GA(4) or higher)
- [ ] Advanced GA operations (full geometric product, rotors, versors)
- [ ] Hybrid architectures (GA layers + standard CNNs)
- [ ] Quantitative explainability metrics
- [ ] Clinical validation with expert evaluation
- [ ] Multi-class classification (beyond binary)
- [ ] Transfer learning for medical imaging

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

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📖 References

### Geometric Algebra
- Hestenes, D. (1986). *New Foundations for Classical Mechanics*
- Dorst, L., Fontijne, D., & Mann, S. (2007). *Geometric Algebra for Computer Science*

### Explainable AI
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks
- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions

### Medical Imaging AI
- Litjens, G., et al. (2017). A survey on deep learning in medical image analysis
- Esteva, A., et al. (2017). Dermatologist-level classification of skin cancer

### Baseline Methods
- DarkCovidNet (MDPI Review)
- VGG-19 for COVID-19 classification
- VGG16 + cGAN (Electronics 2022): "Machine-Learning-Based COVID-19 Detection with Enhanced cGAN Technique Using X-ray Images"

### Dataset
- Cohen, J. P., et al. (2020). COVID-19 image data collection. arXiv:2003.11597

---

**Note**: This system is designed for research and education. For real clinical use, additional validations and appropriate certifications are required.

**Project Status**: Active development - See issues and pull requests for latest updates.
