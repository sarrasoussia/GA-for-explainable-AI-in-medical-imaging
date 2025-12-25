"""
Exemple d'utilisation du système GA pour l'IA explicable en imagerie médicale.
"""

import torch
import numpy as np
from ga_medical_imaging.model import GAMedicalClassifier
from ga_medical_imaging.data_utils import create_dummy_dataset, MedicalImageDataset
from ga_medical_imaging.explainability import GAExplainabilityAnalyzer
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


def example_training():
    """Exemple d'entraînement du modèle."""
    print("=" * 60)
    print("EXEMPLE 1: Entraînement du modèle")
    print("=" * 60)
    
    # Créer un dataset factice
    print("\n1. Création d'un dataset factice...")
    image_paths, labels = create_dummy_dataset(
        num_samples=100,
        image_size=(224, 224),
        output_dir='data/dummy'
    )
    print(f"   ✓ {len(image_paths)} images créées")
    
    # Split train/val
    split_idx = int(len(image_paths) * 0.8)
    train_paths = image_paths[:split_idx]
    train_labels = labels[:split_idx]
    val_paths = image_paths[split_idx:]
    val_labels = labels[split_idx:]
    
    # Créer les datasets
    print("\n2. Préparation des datasets...")
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_dataset = MedicalImageDataset(
        train_paths, train_labels, train_transform, (224, 224)
    )
    val_dataset = MedicalImageDataset(
        val_paths, val_labels, val_transform, (224, 224)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Créer le modèle
    print("\n3. Création du modèle GA...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GAMedicalClassifier(
        num_classes=2,
        multivector_dim=8,
        feature_dim=128,
        device=device
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Modèle créé avec {num_params:,} paramètres")
    
    # Entraînement rapide (quelques époques pour la démo)
    print("\n4. Entraînement du modèle (5 époques pour la démo)...")
    from ga_medical_imaging.train import train
    
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,  # Juste pour la démo
        learning_rate=0.001,
        device=device,
        save_dir='checkpoints',
        use_tensorboard=False
    )
    
    return model, device


def example_explainability(model, device):
    """Exemple d'utilisation du module d'explicabilité."""
    print("\n\n" + "=" * 60)
    print("EXEMPLE 2: Analyse d'explicabilité")
    print("=" * 60)
    
    # Créer une image de test
    print("\n1. Création d'une image de test...")
    test_image = np.random.rand(224, 224) * 0.3
    test_image += np.random.normal(0, 0.1, (224, 224))
    
    # Ajouter une région anormale (simulant une tumeur)
    center_x, center_y = 112, 112
    radius = 30
    y, x = np.ogrid[:224, :224]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    test_image[mask] += 0.5
    test_image = np.clip(test_image, 0, 1)
    
    # Convertir en tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    image_tensor = transform((test_image * 255).astype(np.uint8)).unsqueeze(0).to(device)
    
    # Prédiction
    print("\n2. Prédiction du modèle...")
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    
    class_names = ['Sain', 'Tumeur']
    print(f"   Prédiction: {class_names[pred_class]}")
    print(f"   Probabilités: Sain={probs[0,0].item():.3f}, Tumeur={probs[0,1].item():.3f}")
    
    # Analyse d'explicabilité
    print("\n3. Analyse des composantes géométriques...")
    analyzer = GAExplainabilityAnalyzer(model, device)
    analysis = analyzer.analyze_geometric_components(image_tensor)
    
    # Afficher le rapport
    report = analyzer.generate_explanation_report(image_tensor, class_names=class_names)
    print(report)
    
    # Visualisation
    print("\n4. Génération des visualisations...")
    fig = analyzer.visualize_explanations(
        test_image,
        analysis,
        save_path='explanations/example_explanation.png'
    )
    print("   ✓ Visualisations sauvegardées dans 'explanations/example_explanation.png'")
    
    return analysis


def example_component_analysis(model, device):
    """Exemple d'analyse détaillée des composantes."""
    print("\n\n" + "=" * 60)
    print("EXEMPLE 3: Analyse détaillée des composantes géométriques")
    print("=" * 60)
    
    # Créer une image de test
    test_image = np.random.rand(224, 224) * 0.3
    test_image += np.random.normal(0, 0.1, (224, 224))
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    image_tensor = transform((test_image * 255).astype(np.uint8)).unsqueeze(0).to(device)
    
    # Obtenir les composantes multivecteurs
    print("\n1. Extraction des composantes multivecteurs...")
    components = model.get_multivector_components(image_tensor)
    
    print("\n2. Statistiques des composantes:")
    print(f"   Scalaires (intensités):")
    print(f"     - Moyenne: {components['scalars'].mean().item():.4f}")
    print(f"     - Écart-type: {components['scalars'].std().item():.4f}")
    
    print(f"\n   Vecteurs (gradients):")
    vectors_mag = torch.norm(components['vectors'], dim=-1)
    print(f"     - Magnitude moyenne: {vectors_mag.mean().item():.4f}")
    print(f"     - Magnitude max: {vectors_mag.max().item():.4f}")
    
    print(f"\n   Bivecteurs (orientations/textures):")
    bivectors_mag = torch.norm(components['bivectors'], dim=-1)
    print(f"     - Magnitude moyenne: {bivectors_mag.mean().item():.4f}")
    print(f"     - Magnitude max: {bivectors_mag.max().item():.4f}")
    
    print(f"\n   Trivecteur (relations complexes):")
    print(f"     - Moyenne: {components['trivector'].mean().item():.4f}")
    print(f"     - Écart-type: {components['trivector'].std().item():.4f}")


def main():
    """Fonction principale pour exécuter tous les exemples."""
    print("\n" + "=" * 60)
    print("SYSTÈME GA POUR L'IA EXPLICABLE EN IMAGERIE MÉDICALE")
    print("=" * 60)
    
    # Exemple 1: Entraînement
    model, device = example_training()
    
    # Exemple 2: Explicabilité
    analysis = example_explainability(model, device)
    
    # Exemple 3: Analyse des composantes
    example_component_analysis(model, device)
    
    print("\n\n" + "=" * 60)
    print("EXEMPLES TERMINÉS!")
    print("=" * 60)
    print("\nPour utiliser le système avec vos propres données:")
    print("1. Organisez vos images dans data/sain/ et data/tumeur/")
    print("2. Entraînez: python -m ga_medical_imaging.train --data_dir data")
    print("3. Évaluez: python -m ga_medical_imaging.evaluate_and_explain --checkpoint checkpoints/best_model.pth --image <path>")


if __name__ == '__main__':
    main()

