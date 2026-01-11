# Model Comparison Guide

This guide explains how to use the `compare_models.py` script to compare the GA-based model with a traditional CNN baseline.

## Overview

The comparison script:
1. **Defines a Traditional CNN** baseline model with similar complexity to the GA model
2. **Trains both models** on the same dataset with identical hyperparameters
3. **Calculates comprehensive metrics**: Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, ROC AUC
4. **Generates visualizations**: Confusion matrices, ROC curves, metrics comparison charts
5. **Provides side-by-side comparison** with summary analysis

## Installation

Make sure you have all dependencies installed:

```bash
pip install -r requirements.txt
pip install seaborn  # For better visualizations
```

## Usage

### Basic Usage

Train and compare both models:

```bash
python compare_models.py --num_epochs 30
```

### With Custom Data Directory

```bash
python compare_models.py \
    --data_dir data \
    --num_epochs 50 \
    --batch_size 16
```

### Skip Training (Evaluate Only)

If you already have trained models:

```bash
python compare_models.py \
    --skip_training \
    --output_dir results/comparison
```

## Command Line Arguments

- `--data_dir`: Data directory (if None, creates dummy dataset)
- `--num_epochs`: Number of training epochs (default: 30)
- `--batch_size`: Batch size (default: 16)
- `--learning_rate`: Learning rate (default: 0.001)
- `--device`: Device to use (cpu, cuda, or auto)
- `--output_dir`: Output directory for results (default: results/comparison)
- `--skip_training`: Skip training, only evaluate (requires checkpoints)

## Output Files

The script generates the following outputs in the specified `output_dir`:

### 1. Visualizations

- **`confusion_matrix_ga.png`**: Confusion matrix for GA model
- **`confusion_matrix_cnn.png`**: Confusion matrix for CNN model
- **`roc_curves.png`**: ROC curves comparison
- **`metrics_comparison.png`**: Bar chart comparing all metrics side-by-side

### 2. Data Files

- **`comparison_results.json`**: Complete metrics and model information in JSON format

### 3. Console Output

- Formatted comparison table
- Summary of strengths and weaknesses for each model

## Evaluation Metrics Explained

### Accuracy
Overall correctness: (TP + TN) / (TP + TN + FP + FN)

### Precision
Of all positive predictions, how many were correct: TP / (TP + FP)

### Recall (Sensitivity)
Of all actual positives, how many were found: TP / (TP + FN)

### Specificity
Of all actual negatives, how many were correctly identified: TN / (TN + FP)

### F1-Score
Harmonic mean of precision and recall: 2 × (Precision × Recall) / (Precision + Recall)

### ROC AUC
Area under the Receiver Operating Characteristic curve. Measures the model's ability to distinguish between classes.

## Model Architectures

### GA Model (GAMedicalClassifier)
- Geometric Algebra representation (multivectors)
- GA-based feature extraction
- Maintains geometric structure throughout

### Traditional CNN
- Standard convolutional layers
- Batch normalization
- Global average pooling
- Similar parameter count to GA model

## Interpreting Results

### When GA Model Performs Better

Look for:
- Higher accuracy, F1-score, or ROC AUC
- Better balance between precision and recall
- More stable performance across metrics

**Interpretation**: The GA representation may be capturing geometric features that are more relevant for medical imaging tasks.

### When CNN Performs Better

Look for:
- Higher accuracy or ROC AUC
- Better sensitivity (recall) for detecting tumors

**Interpretation**: Traditional CNNs may benefit from more training data or the specific dataset characteristics. However, GA model still provides intrinsic explainability.

### Key Insight

Even if CNN has slightly higher accuracy, the GA model provides:
- **Intrinsic explainability**: Explanations based on geometric components
- **Geometric interpretability**: Understanding in terms of orientation, magnitude, subspaces
- **Structure-preserving explanations**: Not post-hoc approximations

## Example Output

```
================================================================================
COMPREHENSIVE MODEL COMPARISON
================================================================================
Metric                      GA Model  Traditional CNN
--------------------------------------------------------------------------------
Accuracy                       0.8750           0.8500
Precision                      0.8800           0.8600
Recall (Sensitivity)          0.8700           0.8400
Specificity                    0.8800           0.8600
F1-Score                       0.8750           0.8500
ROC AUC                        0.9200           0.9100
================================================================================
```

## Troubleshooting

### Issue: Models have very different accuracies

**Solution**: 
- Ensure both models are trained for the same number of epochs
- Check that the same data splits are used
- Verify hyperparameters are identical

### Issue: ROC AUC is 0.0

**Solution**:
- This happens when there's only one class in the validation set
- Use a larger, balanced dataset
- Check data distribution

### Issue: Out of memory

**Solution**:
- Reduce batch size: `--batch_size 8`
- Use CPU: `--device cpu`
- Reduce image size in data preparation

## Next Steps

After running the comparison:

1. **Analyze the visualizations** to understand where each model excels
2. **Review the confusion matrices** to see error patterns
3. **Compare ROC curves** to assess discrimination ability
4. **Document findings** in your research report

## Integration with Research Report

Use the results from this comparison to fill in:
- Section 6.1: Performance Metrics Table in `RESEARCH_REPORT.md`
- Section 6.4: Explanation Comparison (add CNN + Grad-CAM comparison)

## Citation

When reporting results, use the framing from `RESEARCH_REPORT.md`:

> "The GA-based model achieves comparable or higher accuracy while offering intrinsically interpretable representations. Unlike traditional post-hoc XAI techniques, the proposed approach embeds interpretability directly into the model's representation."

---

For more details, see:
- `RESEARCH_REPORT.md`: Comprehensive research methodology
- `QUICK_REFERENCE.md`: Quick reference guide

