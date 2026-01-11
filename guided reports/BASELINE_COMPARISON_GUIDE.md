# Baseline Comparison Guide

This guide explains how to evaluate your GA model using the same evaluation protocol as the baseline methods (DarkCovidNet and VGG-19) and compare the results.

## Overview

The baseline methods reported the following results:

### DarkCovidNet (MDPI Review)
- **Dataset**: ieee8023/covid-chestxray-dataset
- **Task**: Binary classification (COVID-19 vs no-findings)
- **Evaluation**: 5-fold Cross-Validation
- **Results**:
  - Accuracy: ≈ 98%
  - Sensitivity: ≈ 95%
  - Specificity: ≈ 95%
  - Precision: ≈ 98%
  - F1-Score: ≈ 0.97

### VGG-19
- **Dataset**: ieee8023/covid-chestxray-dataset + internet-sourced chest X-rays
- **Task**: 2-class (COVID vs non-COVID)
- **Results**:
  - Accuracy: ≈ 98.75%

## Evaluation Protocol

To ensure fair comparison, we use **5-fold cross-validation** matching the DarkCovidNet protocol.

### Metrics Reported

All metrics match the baseline reporting format:
- **Accuracy**: Overall classification accuracy
- **Sensitivity** (Recall): True Positive Rate (TP / (TP + FN))
- **Specificity**: True Negative Rate (TN / (TN + FP))
- **Precision**: Positive Predictive Value (TP / (TP + FP))
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the ROC curve (if available)

## Usage

### Step 1: Run 5-Fold Cross-Validation

First, evaluate your GA model using 5-fold cross-validation:

```bash
python evaluate_5fold_cv.py \
    --data_dir data/covid_chestxray \
    --num_epochs 30 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --output_dir results/5fold_cv
```

**Arguments**:
- `--data_dir`: Path to your dataset directory (organized as `sain/` and `tumeur/` or `covid/` and `no_findings/`)
- `--num_epochs`: Number of training epochs per fold (default: 30)
- `--batch_size`: Batch size (default: 16)
- `--learning_rate`: Learning rate (default: 0.001)
- `--device`: Device to use ('cpu', 'cuda', or 'auto')
- `--output_dir`: Directory to save results (default: `results/5fold_cv`)
- `--n_splits`: Number of CV folds (default: 5)

**Output**:
- `cv_results.json`: Detailed results for each fold and summary statistics

### Step 2: Compare with Baselines

After running cross-validation, compare your results with the baseline methods:

```bash
python compare_with_baselines.py \
    --cv_results results/5fold_cv/cv_results.json \
    --output_dir results/comparison
```

**Arguments**:
- `--cv_results`: Path to the CV results JSON file from Step 1
- `--output_dir`: Directory to save comparison results (default: `results/comparison`)
- `--baselines_file`: Optional JSON file with custom baseline results

**Output**:
- `baseline_comparison.png`: Visualization comparing all metrics
- `comparison_report.txt`: Text report with detailed comparison
- `baseline_comparison.json`: JSON file with all comparison data

### Complete Workflow

For a complete evaluation and comparison:

```bash
# 1. Run 5-fold CV
python evaluate_5fold_cv.py \
    --data_dir data/covid_chestxray \
    --num_epochs 30 \
    --output_dir results/5fold_cv

# 2. Compare with baselines
python compare_with_baselines.py \
    --cv_results results/5fold_cv/cv_results.json \
    --output_dir results/comparison
```

## Dataset Preparation

### Expected Directory Structure

```
data/
└── covid_chestxray/
    ├── covid/          # COVID-19 positive images
    │   ├── image1.png
    │   ├── image2.png
    │   └── ...
    └── no_findings/    # Normal/negative images
        ├── image1.png
        ├── image2.png
        └── ...
```

Or using the French labels:

```
data/
└── covid_chestxray/
    ├── tumeur/         # COVID-19 positive (label=1)
    └── sain/           # Normal/negative (label=0)
```

### Using the COVID-19 Chest X-Ray Dataset

If you're using the `ieee8023/covid-chestxray-dataset`:

1. Download the dataset from GitHub
2. Organize images into `covid/` and `no_findings/` directories
3. Ensure images are in supported formats (PNG, JPG, etc.)

## Understanding the Results

### Cross-Validation Output

The 5-fold CV script outputs:
- **Per-fold results**: Metrics for each fold
- **Summary statistics**: Mean, standard deviation, min, max across folds

Example output:
```
CROSS-VALIDATION SUMMARY (5-Fold)
======================================================================
Metric           Mean         Std          Min          Max
----------------------------------------------------------------------
Accuracy         0.9800       0.0120       0.9650       0.9950
Sensitivity      0.9500       0.0200       0.9200       0.9800
Specificity      0.9600       0.0150       0.9400       0.9800
Precision        0.9750       0.0100       0.9600       0.9900
F1_score         0.9625       0.0125       0.9450       0.9800
Roc_auc          0.9850       0.0080       0.9700       0.9950
======================================================================
```

### Comparison Output

The comparison script generates:
1. **Console table**: Side-by-side comparison of all metrics
2. **Visualization**: Bar charts comparing GA model vs baselines
3. **Text report**: Detailed analysis and key findings

## Interpreting Results

### Performance Comparison

- **Accuracy**: Compare overall classification performance
- **Sensitivity**: Important for medical applications (detecting true positives)
- **Specificity**: Important for avoiding false alarms
- **F1-Score**: Balanced metric considering both precision and recall

### Key Advantages of GA Model

Even if accuracy is comparable, the GA model provides:
1. **Intrinsic Explainability**: Geometric component analysis
2. **Interpretable Features**: Scalars, vectors, bivectors, trivectors
3. **Structure-Preserving**: Explanations reflect actual model computation

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `--batch_size` or use smaller images
2. **Slow Training**: Reduce `--num_epochs` for testing, use GPU if available
3. **Dataset Not Found**: Check `--data_dir` path, or use dummy dataset (omit `--data_dir`)

### Using Dummy Dataset for Testing

To test the evaluation pipeline without real data:

```bash
# This will create a synthetic dataset automatically
python evaluate_5fold_cv.py \
    --num_epochs 10 \
    --output_dir results/5fold_cv_test
```

## Custom Baseline Results

To compare with additional baselines, create a JSON file:

```json
{
  "CustomBaseline": {
    "accuracy": 0.975,
    "sensitivity": 0.94,
    "specificity": 0.96,
    "precision": 0.97,
    "f1_score": 0.955,
    "roc_auc": null,
    "cv_method": "5-fold CV",
    "dataset": "Custom dataset",
    "task": "Binary classification"
  }
}
```

Then use:
```bash
python compare_with_baselines.py \
    --cv_results results/5fold_cv/cv_results.json \
    --baselines_file custom_baselines.json
```

## Next Steps

After obtaining your results:

1. **Document findings**: Update your research report with comparison results
2. **Analyze differences**: Understand why metrics differ (if they do)
3. **Highlight advantages**: Emphasize explainability benefits of GA approach
4. **Statistical testing**: Consider significance tests if needed

## References

- DarkCovidNet: MDPI review paper
- VGG-19: COVID-19 classification studies
- Dataset: ieee8023/covid-chestxray-dataset (GitHub)

---

**Note**: Ensure your dataset and evaluation protocol match the baselines as closely as possible for fair comparison. Differences in preprocessing, data splits, or evaluation metrics can affect comparability.

