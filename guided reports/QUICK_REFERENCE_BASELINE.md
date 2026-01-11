# Quick Reference: Baseline Comparison

## Quick Start

### Complete Evaluation (One Command)

```bash
python run_baseline_comparison.py --data_dir data/covid_chestxray
```

### Step-by-Step

```bash
# 1. Run 5-fold CV
python evaluate_5fold_cv.py --data_dir data/covid_chestxray --output_dir results/5fold_cv

# 2. Compare with baselines
python compare_with_baselines.py --cv_results results/5fold_cv/cv_results.json
```

## Baseline Results (Target)

| Method | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|--------|----------|-------------|-------------|-----------|----------|
| **DarkCovidNet** | 98% | 95% | 95% | 98% | 0.97 |
| **VGG-19** | 98.75% | - | - | - | - |

## Metrics Explained

- **Accuracy**: Overall correctness
- **Sensitivity** (Recall): True Positive Rate (detecting COVID correctly)
- **Specificity**: True Negative Rate (detecting normal correctly)
- **Precision**: Positive Predictive Value
- **F1-Score**: Balanced metric

## Output Files

After running evaluation:

```
results/
├── 5fold_cv/
│   └── cv_results.json          # Detailed CV results
└── comparison/
    ├── baseline_comparison.png  # Visualization
    ├── comparison_report.txt    # Text report
    └── baseline_comparison.json # JSON data
```

## Key Points

✅ **5-fold CV** matches DarkCovidNet protocol  
✅ **All metrics** match baseline reporting format  
✅ **Fair comparison** with same evaluation method  

## Troubleshooting

- **No data?** Omit `--data_dir` to use dummy dataset
- **Out of memory?** Reduce `--batch_size`
- **Slow?** Reduce `--num_epochs` for testing

See `BASELINE_COMPARISON_GUIDE.md` for detailed documentation.

