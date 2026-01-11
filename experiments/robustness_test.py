"""
Script to test model robustness under geometric transformations.
Implements the experimental framework described in RESEARCH_REPORT.md Section 5.2
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from ga_medical_imaging.model import GAMedicalClassifier
from ga_medical_imaging.explainability import GAExplainabilityAnalyzer
from ga_medical_imaging.data_utils import MedicalImageDataset


class TransformedDataset(Dataset):
    """Dataset that applies transformations to base images."""
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if self.transform:
            # Convert tensor to PIL for transformation
            if isinstance(image, torch.Tensor):
                image = transforms.ToPILImage()(image)
            image = self.transform(image)
        return image, label


def apply_rotation(image, angle):
    """Apply rotation transformation."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation([angle, angle]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image)


def apply_scaling(image, scale):
    """Apply scaling transformation."""
    h, w = image.size
    new_h, new_w = int(h * scale), int(w * scale)
    transform = transforms.Compose([
        transforms.Resize((new_h, new_w)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image)


def apply_noise(image, noise_level):
    """Apply Gaussian noise."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    tensor = transform(image)
    noise = torch.randn_like(tensor) * noise_level
    return tensor + noise


def evaluate_robustness(model, dataloader, device, model_name="Model"):
    """Evaluate model accuracy on transformed data."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Evaluating {model_name}"):
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


def test_rotation_robustness(model, base_dataset, device, angles, model_name):
    """Test robustness to rotation."""
    results = {}
    
    print(f"\nTesting rotation robustness for {model_name}...")
    for angle in angles:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation([angle, angle]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        transformed_dataset = TransformedDataset(base_dataset, transform)
        dataloader = DataLoader(transformed_dataset, batch_size=16, shuffle=False)
        
        accuracy = evaluate_robustness(model, dataloader, device, model_name)
        results[angle] = accuracy
        print(f"  Rotation {angle}°: Accuracy = {accuracy:.4f}")
    
    return results


def test_scaling_robustness(model, base_dataset, device, scales, model_name):
    """Test robustness to scaling."""
    results = {}
    
    print(f"\nTesting scaling robustness for {model_name}...")
    for scale in scales:
        def scale_transform(image):
            h, w = image.size
            new_h, new_w = int(h * scale), int(w * scale)
            transform = transforms.Compose([
                transforms.Resize((new_h, new_w)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            return transform(image)
        
        transformed_dataset = TransformedDataset(base_dataset, scale_transform)
        dataloader = DataLoader(transformed_dataset, batch_size=16, shuffle=False)
        
        accuracy = evaluate_robustness(model, dataloader, device, model_name)
        results[scale] = accuracy
        print(f"  Scale {scale}x: Accuracy = {accuracy:.4f}")
    
    return results


def test_noise_robustness(model, base_dataset, device, noise_levels, model_name):
    """Test robustness to noise."""
    results = {}
    
    print(f"\nTesting noise robustness for {model_name}...")
    for noise_level in noise_levels:
        def noise_transform(image):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            tensor = transform(image)
            noise = torch.randn_like(tensor) * noise_level
            return tensor + noise
        
        transformed_dataset = TransformedDataset(base_dataset, noise_transform)
        dataloader = DataLoader(transformed_dataset, batch_size=16, shuffle=False)
        
        accuracy = evaluate_robustness(model, dataloader, device, model_name)
        results[noise_level] = accuracy
        print(f"  Noise σ={noise_level}: Accuracy = {accuracy:.4f}")
    
    return results


def compute_explanation_consistency(model, base_dataset, device, transformations, model_name):
    """Compute explanation consistency across transformations."""
    analyzer = GAExplainabilityAnalyzer(model, device)
    
    # Get a sample image
    sample_image, _ = base_dataset[0]
    if isinstance(sample_image, torch.Tensor):
        sample_image = transforms.ToPILImage()(sample_image)
    
    # Original explanation
    original_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    original_tensor = original_transform(sample_image).unsqueeze(0).to(device)
    original_analysis = analyzer.analyze_geometric_components(original_tensor)
    original_importance = original_analysis['spatial_importance'][0]
    
    consistency_scores = {}
    
    for transform_name, transform_func in transformations.items():
        # Apply transformation
        transformed_tensor = transform_func(sample_image).unsqueeze(0).to(device)
        transformed_analysis = analyzer.analyze_geometric_components(transformed_tensor)
        transformed_importance = transformed_analysis['spatial_importance'][0]
        
        # Compute IoU-like consistency (correlation)
        # Resize to same size if needed
        if original_importance.shape != transformed_importance.shape:
            from scipy.ndimage import zoom
            zoom_factors = [s1/s2 for s1, s2 in zip(original_importance.shape, transformed_importance.shape)]
            transformed_importance = zoom(transformed_importance, zoom_factors)
        
        # Normalize
        original_norm = (original_importance - original_importance.min()) / (original_importance.max() - original_importance.min() + 1e-8)
        transformed_norm = (transformed_importance - transformed_importance.min()) / (transformed_importance.max() - transformed_importance.min() + 1e-8)
        
        # Correlation coefficient
        correlation = np.corrcoef(original_norm.flatten(), transformed_norm.flatten())[0, 1]
        consistency_scores[transform_name] = correlation
    
    return consistency_scores


def plot_robustness_results(all_results, output_dir):
    """Plot robustness comparison charts."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Rotation robustness
    if 'rotation' in all_results:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for model_name, results in all_results['rotation'].items():
            angles = sorted(results.keys())
            accuracies = [results[a] for a in angles]
            ax.plot(angles, accuracies, marker='o', label=model_name)
        ax.set_xlabel('Rotation Angle (degrees)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Rotation Robustness')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'rotation_robustness.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # Scaling robustness
    if 'scaling' in all_results:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for model_name, results in all_results['scaling'].items():
            scales = sorted(results.keys())
            accuracies = [results[s] for s in scales]
            ax.plot(scales, accuracies, marker='o', label=model_name)
        ax.set_xlabel('Scale Factor')
        ax.set_ylabel('Accuracy')
        ax.set_title('Scaling Robustness')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'scaling_robustness.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # Noise robustness
    if 'noise' in all_results:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for model_name, results in all_results['noise'].items():
            noise_levels = sorted(results.keys())
            accuracies = [results[n] for n in noise_levels]
            ax.plot(noise_levels, accuracies, marker='o', label=model_name)
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Noise Robustness')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'noise_robustness.png'), dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test model robustness under transformations')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing test dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu, cuda, or auto)')
    parser.add_argument('--output_dir', type=str, default='results/robustness',
                       help='Output directory for results')
    parser.add_argument('--test_rotation', action='store_true',
                       help='Test rotation robustness')
    parser.add_argument('--test_scaling', action='store_true',
                       help='Test scaling robustness')
    parser.add_argument('--test_noise', action='store_true',
                       help='Test noise robustness')
    parser.add_argument('--test_all', action='store_true',
                       help='Test all transformations')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = GAMedicalClassifier(
        num_classes=2,
        multivector_dim=8,
        feature_dim=128,
        device=device
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {args.checkpoint}")
    
    # Load base dataset
    print("\nLoading dataset...")
    from ga_medical_imaging.data_utils import load_dataset_from_directory
    _, test_loader = load_dataset_from_directory(args.data_dir, image_size=(224, 224))
    base_dataset = test_loader.dataset
    
    # Baseline accuracy (no transformation)
    print("\nComputing baseline accuracy...")
    baseline_acc = evaluate_robustness(model, test_loader, device, "GA Model")
    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    
    all_results = {}
    
    # Test rotation
    if args.test_rotation or args.test_all:
        angles = [0, 15, 30, 45, 60, 90, 135, 180]
        rotation_results = test_rotation_robustness(
            model, base_dataset, device, angles, "GA Model"
        )
        all_results['rotation'] = {'GA Model': rotation_results}
    
    # Test scaling
    if args.test_scaling or args.test_all:
        scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        scaling_results = test_scaling_robustness(
            model, base_dataset, device, scales, "GA Model"
        )
        all_results['scaling'] = {'GA Model': scaling_results}
    
    # Test noise
    if args.test_noise or args.test_all:
        noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
        noise_results = test_noise_robustness(
            model, base_dataset, device, noise_levels, "GA Model"
        )
        all_results['noise'] = {'GA Model': noise_results}
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    results_summary = {
        'baseline_accuracy': baseline_acc,
        'transformations': all_results
    }
    
    with open(os.path.join(args.output_dir, 'robustness_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Plot results
    if all_results:
        plot_robustness_results(all_results, args.output_dir)
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

