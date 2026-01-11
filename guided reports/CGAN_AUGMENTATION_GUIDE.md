# cGAN-Based Data Augmentation Guide

## Overview

This guide explains how to use **Conditional GAN (cGAN)** to generate synthetic COVID-19 chest X-ray images for class balancing, following the approach from:

> **"Machine-Learning-Based COVID-19 Detection with Enhanced cGAN Technique Using X-ray Images"**  
> (Electronics 2022)

## Why Use cGAN Augmentation?

The COVID-19 chest X-ray dataset has severe class imbalance:
- **COVID-19**: 575 images (96.3%)
- **No Finding**: 22 images (3.7%)

The Electronics 2022 paper shows that using cGAN-generated synthetic images to balance classes improves classification performance (accuracy ~99.76% with VGG16).

## Approach

1. **Train a cGAN** on the original dataset to learn the distribution of COVID-19 chest X-rays
2. **Generate synthetic images** to balance the class distribution
3. **Combine** real and synthetic images for training
4. **Evaluate** on real images only (synthetic images are only for training)

## Complete Workflow

### Step 1: Train the cGAN

Train a conditional GAN to learn how to generate COVID-19 chest X-ray images:

```bash
python train_cgan.py \
    --data_dir data/covid_chestxray \
    --num_epochs 100 \
    --batch_size 32 \
    --target_class 1 \
    --save_dir checkpoints/cgan
```

**Parameters**:
- `--data_dir`: Path to your dataset
- `--num_epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--target_class`: Class to generate (1 = COVID-19, 0 = No Finding)
- `--save_dir`: Directory to save checkpoints

**Training Time**: ~2-4 hours on GPU, ~8-12 hours on CPU (for 100 epochs)

### Step 2: Generate Synthetic Images

Generate synthetic images and create a balanced dataset:

```bash
python generate_synthetic_dataset.py \
    --original_data_dir data/covid_chestxray \
    --output_dir data/covid_chestxray_balanced \
    --cgan_checkpoint checkpoints/cgan/cgan_final.pth \
    --target_class 1 \
    --balance_ratio 1.0
```

**Parameters**:
- `--original_data_dir`: Original unbalanced dataset
- `--output_dir`: Output directory for balanced dataset
- `--cgan_checkpoint`: Path to trained cGAN generator
- `--target_class`: Class to generate (1 = COVID-19)
- `--balance_ratio`: 1.0 = fully balanced, 0.5 = half way

This will:
1. Copy all original images
2. Generate synthetic COVID-19 images to match the "No Finding" class size
3. Create a new balanced dataset directory

### Step 3: Evaluate with Balanced Dataset

Run 5-fold cross-validation on the balanced dataset:

```bash
python evaluate_5fold_cv.py \
    --data_dir data/covid_chestxray_balanced \
    --num_epochs 30 \
    --output_dir results/5fold_cv_balanced
```

### Step 4: Compare Results

Compare results with and without cGAN augmentation:

```bash
# Compare with baselines
python compare_with_baselines.py \
    --cv_results results/5fold_cv_balanced/cv_results.json \
    --output_dir results/comparison_balanced
```

## Architecture Details

### Generator Network
- **Input**: Random noise (100-dim) + class label embedding (50-dim)
- **Architecture**: Transposed convolutions with batch normalization
- **Output**: 224×224 grayscale chest X-ray images

### Discriminator Network
- **Input**: Image (224×224) + class label embedding
- **Architecture**: Convolutional layers with batch normalization
- **Output**: Probability that image is real

### Training Strategy
- **Discriminator**: Trained on both real and fake images
- **Generator**: Trained to fool the discriminator
- **Loss Function**: Binary Cross-Entropy with label smoothing
- **Optimizer**: Adam (lr=0.0002, beta1=0.5)

## Expected Results

Based on the Electronics 2022 paper:

| Method | Accuracy | Notes |
|--------|----------|-------|
| VGG16 (baseline) | ~95-97% | Without augmentation |
| VGG16 + cGAN | ~99.76% | With cGAN augmentation |

**Note**: Results may vary based on:
- Dataset size and quality
- cGAN training quality
- Training hyperparameters
- 5-fold CV protocol

## Tips for Best Results

### 1. cGAN Training
- Train for at least 50-100 epochs
- Monitor generator and discriminator losses
- Save checkpoints periodically
- Use GPU if available (much faster)

### 2. Synthetic Image Quality
- Check generated images visually
- Ensure they look realistic
- If images are blurry or unrealistic, train longer
- Consider adjusting learning rates if training is unstable

### 3. Evaluation
- **Always evaluate on real images only**
- Synthetic images are only for training
- Use stratified cross-validation
- Report metrics on real test set

### 4. Class Balancing
- `balance_ratio=1.0`: Fully balanced (recommended)
- `balance_ratio=0.5`: Partially balanced (if fully balanced causes issues)
- Can also generate for minority class (No Finding) if needed

## Troubleshooting

### Issue: Generated images are blurry
**Solution**: 
- Train cGAN for more epochs
- Adjust learning rates
- Increase discriminator capacity

### Issue: Training is unstable
**Solution**:
- Use label smoothing (already implemented)
- Adjust learning rates
- Train discriminator more than generator (already implemented)

### Issue: Out of memory
**Solution**:
- Reduce batch size
- Reduce image size (if acceptable)
- Use gradient accumulation

### Issue: Generated images don't look like chest X-rays
**Solution**:
- Ensure dataset is properly preprocessed
- Check that images are normalized correctly
- Train for more epochs

## Comparison with Other Methods

| Method | Pros | Cons |
|--------|------|------|
| **cGAN** | Realistic images, learns data distribution | Requires training, computationally expensive |
| **Traditional Augmentation** | Fast, simple | Limited diversity |
| **SMOTE** | Works for tabular data | Not suitable for images |
| **Class Weights** | Simple, no data generation | Doesn't add new information |

## Files Created

- `ga_medical_imaging/cgan_generator.py` - cGAN implementation
- `train_cgan.py` - Training script
- `generate_synthetic_dataset.py` - Dataset generation script
- `checkpoints/cgan/` - Trained cGAN models
- `data/covid_chestxray_balanced/` - Balanced dataset

## References

1. **cGAN Paper**: "Machine-Learning-Based COVID-19 Detection with Enhanced cGAN Technique Using X-ray Images" (Electronics 2022)
2. **Original cGAN**: Mirza & Osindero, "Conditional Generative Adversarial Nets" (2014)
3. **Dataset**: ieee8023/covid-chestxray-dataset

## Next Steps

After generating the balanced dataset:

1. **Compare** results with and without cGAN augmentation
2. **Analyze** if synthetic images improve performance
3. **Report** both approaches in your research
4. **Visualize** some generated images to show quality

---

**Note**: The cGAN approach matches the methodology from the Electronics 2022 paper, ensuring fair comparison with their reported results (~99.76% accuracy).

