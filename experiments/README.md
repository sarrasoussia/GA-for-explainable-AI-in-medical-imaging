# Experimental Framework

This directory contains scripts for comprehensive experimental validation of the GA-based XAI framework for medical imaging.

## Scripts

### 1. `compare_baselines.py`

Compares the GA model against baseline methods (ResNet, EfficientNet) on standard classification metrics.

**Usage**:
```bash
python experiments/compare_baselines.py \
    --data_dir data \
    --num_epochs 30 \
    --models ga resnet efficientnet \
    --output_dir results/baseline_comparison
```

**Outputs**:
- `results.json`: Comprehensive metrics (accuracy, precision, recall, F1, AUC-ROC, AUC-PR)
- `metrics_comparison.png`: Bar chart comparison
- `roc_curves.png`: ROC curve comparison

### 2. `robustness_test.py`

Tests model robustness under geometric transformations (rotation, scaling, noise).

**Usage**:
```bash
python experiments/robustness_test.py \
    --data_dir data \
    --checkpoint checkpoints/best_model.pth \
    --test_all \
    --output_dir results/robustness
```

**Options**:
- `--test_rotation`: Test rotation robustness (0° to 180°)
- `--test_scaling`: Test scaling robustness (0.5x to 2.0x)
- `--test_noise`: Test noise robustness (σ = 0.0 to 0.3)
- `--test_all`: Test all transformations

**Outputs**:
- `robustness_results.json`: Accuracy under each transformation
- `rotation_robustness.png`: Rotation vs. accuracy plot
- `scaling_robustness.png`: Scaling vs. accuracy plot
- `noise_robustness.png`: Noise level vs. accuracy plot

## Experimental Workflow

### Step 1: Baseline Comparison

```bash
# Train and compare all models
python experiments/compare_baselines.py \
    --data_dir data \
    --num_epochs 50 \
    --models ga resnet efficientnet
```

### Step 2: Robustness Testing

```bash
# Test GA model robustness
python experiments/robustness_test.py \
    --data_dir data \
    --checkpoint checkpoints/ga_baseline.pth \
    --test_all
```

### Step 3: Analysis

Review the generated plots and JSON files to:
- Compare accuracy across models
- Assess robustness under transformations
- Identify strengths and weaknesses

## Expected Results

Based on the research framework:

1. **Accuracy**: GA model should achieve **comparable or higher** accuracy than baselines
2. **Robustness**: GA model should show **smoother degradation** under transformations
3. **Explainability**: GA model provides **intrinsic explanations** (tested separately)

## Notes

- These scripts require trained models. Train first using `python -m ga_medical_imaging.train`
- Results are saved in JSON format for easy analysis
- Plots are generated automatically for visualization

