# COVID-19 Chest X-Ray Dataset Information

## Dataset Source

The dataset is downloaded from: [ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset)

This is the same dataset used by the baseline methods:
- **DarkCovidNet**: Used this dataset for binary classification (COVID-19 vs no-findings)
- **VGG-19**: Used this dataset + internet-sourced chest X-rays

## Dataset Organization

After running `organize_covid_dataset.py`, the dataset is organized as:

```
data/covid_chestxray/
├── covid/              # COVID-19 positive cases (label=1)
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── no_findings/        # Normal/negative cases (label=0)
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Dataset Statistics

From the metadata analysis:
- **COVID-19 positive cases**: ~575 images (after filtering and removing missing files)
- **No Finding cases**: 22 images
- **Total images**: ~597 images

### Class Imbalance

⚠️ **Important Note**: The dataset has significant class imbalance:
- COVID-19: 575 images (96.3%)
- No Finding: 22 images (3.7%)

This imbalance is common in medical datasets and was present in the baseline studies as well.

### Handling Class Imbalance

When training, consider:

1. **cGAN-Based Augmentation** (Recommended - matches Electronics 2022 paper):
   - Train a conditional GAN to generate synthetic COVID-19 images
   - Balance classes by adding synthetic images
   - See `CGAN_AUGMENTATION_GUIDE.md` for details
   - Reported accuracy: ~99.76% with VGG16 + cGAN
   ```bash
   python run_cgan_pipeline.py --data_dir data/covid_chestxray
   ```

2. **Class Weights**: Use weighted loss function
   ```python
   from torch.nn import CrossEntropyLoss
   class_weights = torch.tensor([575/22, 1.0])  # Weight negative class more
   criterion = CrossEntropyLoss(weight=class_weights)
   ```

3. **Data Augmentation**: Apply more aggressive augmentation to the minority class

4. **Oversampling**: Use techniques like SMOTE or random oversampling

5. **Evaluation Metrics**: Focus on metrics that handle imbalance well:
   - F1-Score
   - ROC AUC
   - Precision-Recall AUC
   - Sensitivity and Specificity

6. **Stratified Cross-Validation**: Ensure each fold maintains class distribution

## Usage

### Organize the Dataset

```bash
python organize_covid_dataset.py \
    --dataset_dir data/covid-chestxray-dataset \
    --output_dir data/covid_chestxray
```

### Use with Evaluation Scripts

```bash
# 5-fold Cross-Validation
python evaluate_5fold_cv.py --data_dir data/covid_chestxray

# Complete evaluation and comparison
python run_baseline_comparison.py --data_dir data/covid_chestxray
```

## Dataset Details

### Image Formats
- Supported formats: PNG, JPG, JPEG
- Images are typically chest X-rays (PA or AP views)

### Metadata
The original dataset includes rich metadata in `metadata.csv`:
- Patient demographics (age, sex)
- Clinical findings
- RT-PCR test results
- Survival outcomes
- Intubation status
- And more...

### License
Each image has its own license specified in the metadata.csv file. Common licenses include:
- Apache 2.0
- CC BY-NC-SA 4.0
- CC BY 4.0

The metadata.csv, scripts, and other documents are released under CC BY-NC-SA 4.0 license.

## Citation

If you use this dataset, please cite:

```bibtex
@article{cohen2020covid,
  title={COVID-19 image data collection},
  author={Joseph Paul Cohen and Paul Morrison and Lan Dao},
  journal={arXiv 2003.11597},
  url={https://github.com/ieee8023/covid-chestxray-dataset},
  year={2020}
}
```

## References

- Dataset Repository: https://github.com/ieee8023/covid-chestxray-dataset
- Original Paper: Cohen et al., "COVID-19 image data collection", arXiv:2003.11597, 2020
- Prospective Paper: Cohen et al., "COVID-19 Image Data Collection: Prospective Predictions Are the Future", arXiv:2006.11988, 2020

