"""
Script pour évaluer le modèle et générer des explications.
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from .model import GAMedicalClassifier
from .explainability import GAExplainabilityAnalyzer


def load_model(checkpoint_path: str, device: str = 'cpu') -> nn.Module:
    """Charge un modèle depuis un checkpoint."""
    model = GAMedicalClassifier(
        num_classes=2,
        multivector_dim=8,
        feature_dim=128,
        device=device
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Modèle chargé depuis {checkpoint_path}")
    if 'val_acc' in checkpoint:
        print(f"Précision de validation: {checkpoint['val_acc']:.2f}%")
    
    return model


def evaluate_image(
    model: nn.Module,
    image_path: str,
    device: str = 'cpu',
    save_explanations: bool = True,
    output_dir: str = 'explanations'
):
    """
    Évalue une image et génère des explications.
    
    Args:
        model: Modèle entraîné
        image_path: Chemin vers l'image
        device: Device PyTorch
        save_explanations: Sauvegarder les visualisations
        output_dir: Répertoire de sortie
    """
    # Charger et préprocesser l'image
    image = Image.open(image_path).convert('L')
    original_image = np.array(image)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Prédiction
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    
    class_names = ['Sain', 'Tumeur']
    print(f"\n{'='*60}")
    print(f"ÉVALUATION DE L'IMAGE: {image_path}")
    print(f"{'='*60}")
    print(f"\nPrédiction: {class_names[pred_class]}")
    print(f"Probabilités:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {probs[0, i].item():.3f} ({probs[0, i].item()*100:.1f}%)")
    
    # Analyse d'explicabilité
    analyzer = GAExplainabilityAnalyzer(model, device)
    
    print("\nAnalyse des composantes géométriques...")
    analysis = analyzer.analyze_geometric_components(image_tensor)
    
    # Afficher le rapport
    report = analyzer.generate_explanation_report(
        image_tensor,
        class_names=class_names
    )
    print(report)
    
    # Visualisation
    if save_explanations:
        os.makedirs(output_dir, exist_ok=True)
        
        # Préparer l'image pour la visualisation
        image_for_viz = original_image
        if len(image_for_viz.shape) == 3:
            image_for_viz = np.mean(image_for_viz, axis=2)
        
        # Normaliser pour la visualisation
        image_for_viz = (image_for_viz - image_for_viz.min()) / (image_for_viz.max() - image_for_viz.min() + 1e-8)
        
        # Créer la visualisation
        fig = analyzer.visualize_explanations(
            image_for_viz,
            analysis,
            save_path=os.path.join(
                output_dir,
                f"explanation_{os.path.basename(image_path).split('.')[0]}.png"
            )
        )
        
        print(f"\nVisualisations sauvegardées dans: {output_dir}")
    
    return analysis, pred_class, probs


def main():
    parser = argparse.ArgumentParser(
        description='Évaluer le modèle et générer des explications'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Chemin vers le checkpoint du modèle')
    parser.add_argument('--image', type=str, required=True,
                       help='Chemin vers l\'image à évaluer')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu, cuda, ou auto)')
    parser.add_argument('--output_dir', type=str, default='explanations',
                       help='Répertoire pour sauvegarder les explications')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Charger le modèle
    model = load_model(args.checkpoint, device)
    
    # Évaluer l'image
    evaluate_image(
        model=model,
        image_path=args.image,
        device=device,
        save_explanations=True,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

