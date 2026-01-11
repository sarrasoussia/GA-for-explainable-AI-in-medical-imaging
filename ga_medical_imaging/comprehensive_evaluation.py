"""
Comprehensive Evaluation Framework for GA-based Image Classification

This module implements a multi-dimensional evaluation framework that evaluates
algorithmic properties beyond traditional accuracy metrics:

1. Data Efficiency: Performance with limited training data
2. Robustness: Stability under perturbations (noise, blur, occlusion)
3. Explainability: Intrinsic vs post-hoc explanations (consistency and stability)
4. Failure Transparency: Confidence calibration and error detection
5. Representation Expressiveness: GA multivectors vs pixel tensors

This aligns with the evaluation strategy where "better" means:
- Comparable accuracy (not necessarily higher)
- Better data efficiency
- More robust under distribution shift
- Intrinsic explainability (vs post-hoc)
- Higher failure transparency
- More stable and consistent explanations
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Dict, List, Tuple, Optional
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from scipy.stats import entropy
from scipy.ndimage import zoom
import argparse

from .model import GAMedicalClassifier
from .explainability import GAExplainabilityAnalyzer
from .data_utils import load_dataset_from_directory


class ComprehensiveEvaluator:
    """
    Comprehensive evaluator that measures multiple dimensions of model performance
    beyond traditional accuracy metrics.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        explainability_analyzer: Optional[GAExplainabilityAnalyzer] = None
    ):
        """
        Args:
            model: Trained model to evaluate
            device: Device for computation
            explainability_analyzer: Optional explainability analyzer (for GA models)
        """
        self.model = model
        self.device = device
        self.model.eval()
        self.explainability_analyzer = explainability_analyzer
        
        if explainability_analyzer is None and hasattr(model, 'ga_representation'):
            self.explainability_analyzer = GAExplainabilityAnalyzer(model, device)
    
    def evaluate_standard_metrics(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate standard classification metrics.
        
        Returns:
            Dictionary with accuracy, precision, recall, f1, roc_auc
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        all_logits = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating standard metrics"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(images)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy() if probs.shape[1] > 1 else probs.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_logits = np.array(all_logits)
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'specificity': self._calculate_specificity(all_labels, all_preds),
            'f1_score': f1_score(all_labels, all_preds, zero_division=0),
            'roc_auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
        }
        
        # Calculate AUC-PR
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
            metrics['auc_pr'] = np.trapz(precision_curve, recall_curve)
        except:
            metrics['auc_pr'] = 0.0
        
        return metrics, all_probs, all_labels, all_logits
    
    def _calculate_specificity(self, y_true, y_pred):
        """Calculate specificity (true negative rate)."""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def evaluate_data_efficiency(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        data_fractions: List[float] = [1.0, 0.5, 0.25, 0.1],
        num_epochs: int = 30,
        batch_size: int = 16,
        learning_rate: float = 0.001
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance with different amounts of training data.
        
        This tests data efficiency - GA models should degrade more gracefully
        than CNNs with limited data.
        
        Args:
            train_dataset: Full training dataset
            val_dataset: Validation dataset
            data_fractions: List of fractions of data to use (e.g., [1.0, 0.5, 0.25, 0.1])
            num_epochs: Number of epochs to train
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Dictionary mapping data fraction to metrics
        """
        results = {}
        
        for fraction in data_fractions:
            print(f"\n{'='*60}")
            print(f"Training with {fraction*100:.0f}% of data")
            print(f"{'='*60}")
            
            # Create subset
            n_samples = int(len(train_dataset) * fraction)
            indices = np.random.choice(len(train_dataset), n_samples, replace=False)
            subset = Subset(train_dataset, indices)
            subset_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Create fresh model
            model = GAMedicalClassifier(
                num_classes=2,
                multivector_dim=8,
                feature_dim=128,
                device=self.device
            ).to(self.device)
            
            # Train model
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            best_val_acc = 0.0
            for epoch in range(num_epochs):
                # Training
                model.train()
                train_loss = 0.0
                for images, labels in subset_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    optimizer.zero_grad()
                    logits = model(images)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        logits = model(images)
                        preds = torch.argmax(logits, dim=1)
                        val_correct += (preds == labels).sum().item()
                        val_total += labels.size(0)
                
                val_acc = val_correct / val_total
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{num_epochs}: Val Acc = {val_acc:.4f}")
            
            # Final evaluation
            evaluator = ComprehensiveEvaluator(model, self.device)
            metrics, _, _, _ = evaluator.evaluate_standard_metrics(val_loader)
            metrics['best_val_acc'] = best_val_acc
            
            results[f"{fraction*100:.0f}%"] = metrics
            print(f"  Final Accuracy: {metrics['accuracy']:.4f}")
        
        return results
    
    def evaluate_robustness(
        self,
        dataloader: DataLoader,
        perturbations: Dict[str, callable]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate robustness under various perturbations.
        
        Measures both accuracy degradation and explanation stability.
        
        Args:
            dataloader: Data loader
            perturbations: Dictionary mapping perturbation name to transform function
            
        Returns:
            Dictionary with accuracy and explanation consistency for each perturbation
        """
        results = {}
        
        # Baseline (no perturbation)
        baseline_metrics, _, _, _ = self.evaluate_standard_metrics(dataloader)
        baseline_acc = baseline_metrics['accuracy']
        
        # Get baseline explanations if available
        baseline_explanations = None
        if self.explainability_analyzer is not None:
            # Get a sample batch for explanation consistency
            sample_images, _ = next(iter(dataloader))
            sample_images = sample_images[:4].to(self.device)  # Use first 4 samples
            baseline_analysis = self.explainability_analyzer.analyze_geometric_components(sample_images)
            baseline_explanations = baseline_analysis['spatial_importance']
        
        results['baseline'] = {
            'accuracy': baseline_acc,
            'explanation_consistency': 1.0  # Perfect consistency with itself
        }
        
        # Test each perturbation
        for pert_name, pert_func in perturbations.items():
            print(f"\nTesting perturbation: {pert_name}")
            
            # Apply perturbation to dataset
            perturbed_dataset = self._apply_perturbation_to_dataset(
                dataloader.dataset, pert_func
            )
            perturbed_loader = DataLoader(
                perturbed_dataset, batch_size=dataloader.batch_size, shuffle=False
            )
            
            # Evaluate accuracy
            pert_metrics, _, _, _ = self.evaluate_standard_metrics(perturbed_loader)
            pert_acc = pert_metrics['accuracy']
            acc_drop = baseline_acc - pert_acc
            
            # Evaluate explanation consistency
            explanation_consistency = 1.0
            if self.explainability_analyzer is not None and baseline_explanations is not None:
                perturbed_images = sample_images.clone()
                for i in range(len(perturbed_images)):
                    # Apply perturbation (convert to PIL, transform, convert back)
                    from torchvision import transforms
                    to_pil = transforms.ToPILImage()
                    to_tensor = transforms.ToTensor()
                    pil_img = to_pil(perturbed_images[i].cpu())
                    pert_img = pert_func(pil_img)
                    if isinstance(pert_img, torch.Tensor):
                        perturbed_images[i] = pert_img
                    else:
                        perturbed_images[i] = to_tensor(pert_img)
                
                perturbed_images = perturbed_images.to(self.device)
                pert_analysis = self.explainability_analyzer.analyze_geometric_components(perturbed_images)
                pert_explanations = pert_analysis['spatial_importance']
                
                # Compute consistency (correlation)
                consistency_scores = []
                for i in range(len(baseline_explanations)):
                    base_exp = baseline_explanations[i]
                    pert_exp = pert_explanations[i]
                    
                    # Resize if needed
                    if base_exp.shape != pert_exp.shape:
                        zoom_factors = [s1/s2 for s1, s2 in zip(base_exp.shape, pert_exp.shape)]
                        pert_exp = zoom(pert_exp, zoom_factors)
                    
                    # Normalize
                    base_norm = (base_exp - base_exp.min()) / (base_exp.max() - base_exp.min() + 1e-8)
                    pert_norm = (pert_exp - pert_exp.min()) / (pert_exp.max() - pert_exp.min() + 1e-8)
                    
                    # Correlation
                    correlation = np.corrcoef(base_norm.flatten(), pert_norm.flatten())[0, 1]
                    consistency_scores.append(correlation)
                
                explanation_consistency = np.mean(consistency_scores)
            
            results[pert_name] = {
                'accuracy': pert_acc,
                'accuracy_drop': acc_drop,
                'relative_drop': acc_drop / baseline_acc if baseline_acc > 0 else 0.0,
                'explanation_consistency': explanation_consistency
            }
            
            print(f"  Accuracy: {pert_acc:.4f} (drop: {acc_drop:.4f})")
            print(f"  Explanation Consistency: {explanation_consistency:.4f}")
        
        return results
    
    def _apply_perturbation_to_dataset(self, dataset: Dataset, pert_func: callable):
        """Create a dataset with applied perturbation."""
        class PerturbedDataset(Dataset):
            def __init__(self, base_dataset, pert_func):
                self.base_dataset = base_dataset
                self.pert_func = pert_func
            
            def __len__(self):
                return len(self.base_dataset)
            
            def __getitem__(self, idx):
                image, label = self.base_dataset[idx]
                from torchvision import transforms
                to_pil = transforms.ToPILImage()
                to_tensor = transforms.ToTensor()
                
                if isinstance(image, torch.Tensor):
                    pil_img = to_pil(image)
                else:
                    pil_img = image
                
                pert_img = self.pert_func(pil_img)
                if isinstance(pert_img, torch.Tensor):
                    return pert_img, label
                else:
                    return to_tensor(pert_img), label
        
        return PerturbedDataset(dataset, pert_func)
    
    def evaluate_explainability_metrics(
        self,
        dataloader: DataLoader,
        num_samples: int = 20
    ) -> Dict[str, float]:
        """
        Evaluate explainability quality metrics:
        - Explanation consistency: Stability across perturbations
        - Faithfulness: Correlation between explanation importance and prediction change
        - Localization: Agreement with known regions (if available)
        
        Args:
            dataloader: Data loader
            num_samples: Number of samples to evaluate
            
        Returns:
            Dictionary with explainability metrics
        """
        if self.explainability_analyzer is None:
            return {
                'explanation_consistency': 0.0,
                'faithfulness': 0.0,
                'localization_agreement': 0.0
            }
        
        # Get sample batch
        sample_images, sample_labels = next(iter(dataloader))
        sample_images = sample_images[:num_samples].to(self.device)
        sample_labels = sample_labels[:num_samples]
        
        # 1. Explanation Consistency
        # Test consistency across small perturbations
        consistency_scores = []
        perturbations = [
            ('rotation_5', lambda img: transforms.RandomRotation([5, 5])(img)),
            ('rotation_-5', lambda img: transforms.RandomRotation([-5, -5])(img)),
            ('noise_0.05', lambda img: self._add_noise_to_tensor(img, 0.05))
        ]
        
        from torchvision import transforms
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()
        
        # Get baseline explanations
        baseline_analysis = self.explainability_analyzer.analyze_geometric_components(sample_images)
        baseline_importance = baseline_analysis['spatial_importance']
        
        for pert_name, pert_func in perturbations:
            pert_images = []
            for img in sample_images:
                pil_img = to_pil(img.cpu())
                pert_img = pert_func(pil_img)
                if isinstance(pert_img, torch.Tensor):
                    pert_images.append(pert_img)
                else:
                    pert_images.append(to_tensor(pert_img))
            
            pert_images = torch.stack(pert_images).to(self.device)
            pert_analysis = self.explainability_analyzer.analyze_geometric_components(pert_images)
            pert_importance = pert_analysis['spatial_importance']
            
            # Compute correlation for each sample
            for i in range(len(baseline_importance)):
                base_exp = baseline_importance[i]
                pert_exp = pert_importance[i]
                
                if base_exp.shape != pert_exp.shape:
                    zoom_factors = [s1/s2 for s1, s2 in zip(base_exp.shape, pert_exp.shape)]
                    pert_exp = zoom(pert_exp, zoom_factors)
                
                base_norm = (base_exp - base_exp.min()) / (base_exp.max() - base_exp.min() + 1e-8)
                pert_norm = (pert_exp - pert_exp.min()) / (pert_exp.max() - pert_exp.min() + 1e-8)
                
                correlation = np.corrcoef(base_norm.flatten(), pert_norm.flatten())[0, 1]
                consistency_scores.append(correlation)
        
        explanation_consistency = np.mean(consistency_scores)
        
        # 2. Faithfulness
        # Remove top-k important regions and measure prediction change
        faithfulness_scores = []
        k_values = [0.1, 0.2, 0.3]  # Remove top 10%, 20%, 30% of important regions
        
        with torch.no_grad():
            # Get original predictions
            original_logits = self.model(sample_images)
            original_probs = torch.softmax(original_logits, dim=1)
            original_preds = torch.argmax(original_logits, dim=1)
            
            for k in k_values:
                # Create masked images (remove top-k important regions)
                masked_images = sample_images.clone()
                for i, img in enumerate(sample_images):
                    importance_map = baseline_importance[i]
                    threshold = np.percentile(importance_map, (1 - k) * 100)
                    mask = importance_map < threshold
                    
                    # Apply mask (set to mean)
                    img_np = img[0].cpu().numpy()
                    img_np[mask] = img_np.mean()
                    masked_images[i, 0] = torch.from_numpy(img_np).to(self.device)
                
                # Get predictions on masked images
                masked_logits = self.model(masked_images)
                masked_probs = torch.softmax(masked_logits, dim=1)
                
                # Measure prediction change
                prob_change = torch.abs(original_probs - masked_probs).mean().item()
                faithfulness_scores.append(prob_change)
        
        # Higher change = more faithful (removing important regions changes prediction)
        faithfulness = np.mean(faithfulness_scores)
        
        # 3. Localization Agreement (placeholder - requires ground truth masks)
        # This would compare explanation regions with expert annotations
        localization_agreement = 0.0  # Would need ground truth masks
        
        return {
            'explanation_consistency': float(explanation_consistency),
            'faithfulness': float(faithfulness),
            'localization_agreement': float(localization_agreement)
        }
    
    def _add_noise_to_tensor(self, tensor: torch.Tensor, noise_level: float) -> torch.Tensor:
        """Add Gaussian noise to tensor."""
        noise = torch.randn_like(tensor) * noise_level
        return torch.clamp(tensor + noise, 0, 1)
    
    def evaluate_failure_transparency(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate failure transparency:
        - Confidence calibration: Are high-confidence predictions actually correct?
        - Entropy analysis: Do wrong predictions have higher entropy?
        - Overconfidence detection: Are wrong predictions overconfident?
        
        Args:
            dataloader: Data loader
            
        Returns:
            Dictionary with failure transparency metrics
        """
        self.model.eval()
        all_probs = []
        all_labels = []
        all_preds = []
        all_entropies = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating failure transparency"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(images)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                # Calculate entropy
                entropies = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_entropies.extend(entropies.cpu().numpy())
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_entropies = np.array(all_entropies)
        
        # Confidence for predicted class
        confidences = np.max(all_probs, axis=1)
        
        # Separate correct and incorrect predictions
        correct_mask = (all_preds == all_labels)
        incorrect_mask = ~correct_mask
        
        correct_confidences = confidences[correct_mask]
        incorrect_confidences = confidences[incorrect_mask]
        correct_entropies = all_entropies[correct_mask]
        incorrect_entropies = all_entropies[incorrect_mask]
        
        # Metrics
        metrics = {
            'mean_confidence_correct': float(np.mean(correct_confidences)) if len(correct_confidences) > 0 else 0.0,
            'mean_confidence_incorrect': float(np.mean(incorrect_confidences)) if len(incorrect_confidences) > 0 else 0.0,
            'mean_entropy_correct': float(np.mean(correct_entropies)) if len(correct_entropies) > 0 else 0.0,
            'mean_entropy_incorrect': float(np.mean(incorrect_entropies)) if len(incorrect_entropies) > 0 else 0.0,
            'overconfidence_score': float(np.mean(incorrect_confidences)) if len(incorrect_confidences) > 0 else 0.0,
            'confidence_gap': float(np.mean(correct_confidences) - np.mean(incorrect_confidences)) if len(correct_confidences) > 0 and len(incorrect_confidences) > 0 else 0.0
        }
        
        # Expected: Correct predictions should have higher confidence and lower entropy
        # Wrong predictions should have lower confidence and higher entropy
        # Overconfidence = high confidence on wrong predictions (bad)
        
        return metrics
    
    def generate_comprehensive_report(
        self,
        results: Dict,
        output_path: str
    ):
        """Generate a comprehensive evaluation report."""
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE EVALUATION REPORT")
        report.append("="*80)
        report.append("")
        report.append("Multi-Dimensional Evaluation: Beyond Accuracy")
        report.append("")
        
        # Standard metrics
        if 'standard_metrics' in results:
            report.append("STANDARD METRICS")
            report.append("-"*80)
            metrics = results['standard_metrics']
            for key, value in metrics.items():
                report.append(f"  {key}: {value:.4f}")
            report.append("")
        
        # Data efficiency
        if 'data_efficiency' in results:
            report.append("DATA EFFICIENCY")
            report.append("-"*80)
            report.append("Performance degradation with limited training data:")
            for fraction, metrics in results['data_efficiency'].items():
                report.append(f"  {fraction} of data: Accuracy = {metrics['accuracy']:.4f}")
            report.append("")
        
        # Robustness
        if 'robustness' in results:
            report.append("ROBUSTNESS")
            report.append("-"*80)
            baseline = results['robustness'].get('baseline', {})
            report.append(f"  Baseline Accuracy: {baseline.get('accuracy', 0):.4f}")
            for pert_name, pert_results in results['robustness'].items():
                if pert_name != 'baseline':
                    acc_drop = pert_results.get('accuracy_drop', 0)
                    exp_consistency = pert_results.get('explanation_consistency', 0)
                    report.append(f"  {pert_name}:")
                    report.append(f"    Accuracy Drop: {acc_drop:.4f}")
                    report.append(f"    Explanation Consistency: {exp_consistency:.4f}")
            report.append("")
        
        # Explainability
        if 'explainability' in results:
            report.append("EXPLAINABILITY METRICS")
            report.append("-"*80)
            exp_metrics = results['explainability']
            report.append(f"  Explanation Consistency: {exp_metrics.get('explanation_consistency', 0):.4f}")
            report.append(f"  Faithfulness: {exp_metrics.get('faithfulness', 0):.4f}")
            report.append(f"  Localization Agreement: {exp_metrics.get('localization_agreement', 0):.4f}")
            report.append("")
        
        # Failure transparency
        if 'failure_transparency' in results:
            report.append("FAILURE TRANSPARENCY")
            report.append("-"*80)
            ft_metrics = results['failure_transparency']
            report.append(f"  Mean Confidence (Correct): {ft_metrics.get('mean_confidence_correct', 0):.4f}")
            report.append(f"  Mean Confidence (Incorrect): {ft_metrics.get('mean_confidence_incorrect', 0):.4f}")
            report.append(f"  Mean Entropy (Correct): {ft_metrics.get('mean_entropy_correct', 0):.4f}")
            report.append(f"  Mean Entropy (Incorrect): {ft_metrics.get('mean_entropy_incorrect', 0):.4f}")
            report.append(f"  Overconfidence Score: {ft_metrics.get('overconfidence_score', 0):.4f}")
            report.append(f"  Confidence Gap: {ft_metrics.get('confidence_gap', 0):.4f}")
            report.append("")
        
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        print(f"\nComprehensive report saved to: {output_path}")
        print(report_text)
        
        return report_text


def create_perturbation_functions():
    """Create standard perturbation functions for robustness testing."""
    from torchvision import transforms
    
    perturbations = {}
    
    # Gaussian noise
    def gaussian_noise(noise_level: float):
        def pert_func(image):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            tensor = transform(image)
            noise = torch.randn_like(tensor) * noise_level
            return torch.clamp(tensor + noise, -1, 1)
        return pert_func
    
    perturbations['gaussian_noise_0.1'] = gaussian_noise(0.1)
    perturbations['gaussian_noise_0.2'] = gaussian_noise(0.2)
    
    # Contrast shift
    def contrast_shift(alpha: float):
        def pert_func(image):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            tensor = transform(image)
            return torch.clamp(tensor * alpha, -1, 1)
        return pert_func
    
    perturbations['contrast_high'] = contrast_shift(1.5)
    perturbations['contrast_low'] = contrast_shift(0.5)
    
    # Rotation
    def rotation(angle: float):
        def pert_func(image):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation([angle, angle]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            return transform(image)
        return pert_func
    
    perturbations['rotation_15'] = rotation(15)
    perturbations['rotation_30'] = rotation(30)
    perturbations['rotation_45'] = rotation(45)
    
    return perturbations


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive evaluation framework for GA-based XAI'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results/comprehensive_evaluation',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu, cuda, mps, or auto)')
    parser.add_argument('--test_data_efficiency', action='store_true',
                       help='Test data efficiency (requires training)')
    parser.add_argument('--test_robustness', action='store_true',
                       help='Test robustness')
    parser.add_argument('--test_explainability', action='store_true',
                       help='Test explainability metrics')
    parser.add_argument('--test_failure_transparency', action='store_true',
                       help='Test failure transparency')
    parser.add_argument('--test_all', action='store_true',
                       help='Run all tests')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
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
    
    # Load dataset
    print("\nLoading dataset...")
    train_loader, val_loader = load_dataset_from_directory(
        args.data_dir, image_size=(224, 224)
    )
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(model, device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    
    # Standard metrics (always run)
    print("\n" + "="*60)
    print("EVALUATING STANDARD METRICS")
    print("="*60)
    metrics, _, _, _ = evaluator.evaluate_standard_metrics(val_loader)
    results['standard_metrics'] = metrics
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Data efficiency
    if args.test_data_efficiency or args.test_all:
        print("\n" + "="*60)
        print("EVALUATING DATA EFFICIENCY")
        print("="*60)
        data_efficiency_results = evaluator.evaluate_data_efficiency(
            train_loader.dataset,
            val_loader.dataset,
            data_fractions=[1.0, 0.5, 0.25, 0.1],
            num_epochs=30
        )
        results['data_efficiency'] = data_efficiency_results
    
    # Robustness
    if args.test_robustness or args.test_all:
        print("\n" + "="*60)
        print("EVALUATING ROBUSTNESS")
        print("="*60)
        perturbations = create_perturbation_functions()
        robustness_results = evaluator.evaluate_robustness(val_loader, perturbations)
        results['robustness'] = robustness_results
    
    # Explainability
    if args.test_explainability or args.test_all:
        print("\n" + "="*60)
        print("EVALUATING EXPLAINABILITY METRICS")
        print("="*60)
        explainability_results = evaluator.evaluate_explainability_metrics(val_loader)
        results['explainability'] = explainability_results
    
    # Failure transparency
    if args.test_failure_transparency or args.test_all:
        print("\n" + "="*60)
        print("EVALUATING FAILURE TRANSPARENCY")
        print("="*60)
        failure_transparency_results = evaluator.evaluate_failure_transparency(val_loader)
        results['failure_transparency'] = failure_transparency_results
    
    # Save results
    results_path = os.path.join(args.output_dir, 'comprehensive_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Generate report
    report_path = os.path.join(args.output_dir, 'comprehensive_report.txt')
    evaluator.generate_comprehensive_report(results, report_path)
    
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

