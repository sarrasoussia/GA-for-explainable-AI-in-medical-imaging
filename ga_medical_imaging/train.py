"""
Script d'entraînement pour le modèle GA d'imagerie médicale.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
from typing import Optional
import argparse

from .model import GAMedicalClassifier
from .data_utils import load_dataset_from_directory, create_dummy_dataset, MedicalImageDataset
from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int
) -> float:
    """Entraîne le modèle pour une époque."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistiques
        running_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Mettre à jour la barre de progression
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> tuple:
    """Valide le modèle."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    save_dir: str = 'checkpoints',
    use_tensorboard: bool = True
):
    """
    Fonction principale d'entraînement.
    
    Args:
        model: Modèle à entraîner
        train_loader: DataLoader pour l'entraînement
        val_loader: DataLoader pour la validation
        num_epochs: Nombre d'époques
        learning_rate: Taux d'apprentissage
        device: Device PyTorch
        save_dir: Répertoire pour sauvegarder les checkpoints
        use_tensorboard: Utiliser TensorBoard pour le logging
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Loss et optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # TensorBoard
    writer = None
    if use_tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
    
    best_val_acc = 0.0
    
    print(f"Début de l'entraînement sur {device}")
    print(f"Nombre d'époques: {num_epochs}")
    print(f"Taille du dataset d'entraînement: {len(train_loader.dataset)}")
    print(f"Taille du dataset de validation: {len(val_loader.dataset)}")
    print("-" * 50)
    
    for epoch in range(1, num_epochs + 1):
        # Entraînement
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Scheduler
        scheduler.step(val_loss)
        
        # Logging
        if writer:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        
        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Sauvegarder le meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  ✓ Meilleur modèle sauvegardé (Acc: {val_acc:.2f}%)")
        
        # Checkpoint périodique
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    if writer:
        writer.close()
    
    print("\n" + "=" * 50)
    print(f"Entraînement terminé!")
    print(f"Meilleure précision de validation: {best_val_acc:.2f}%")
    print(f"Modèle sauvegardé dans: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Entraînement du modèle GA pour imagerie médicale')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Répertoire des données (si None, crée un dataset factice)')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Nombre d\'époques')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Taille du batch')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Taux d\'apprentissage')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu, cuda, ou auto)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Répertoire pour sauvegarder les checkpoints')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224],
                       help='Taille des images (H W)')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Device: {device}")
    
    # Charger les données
    if args.data_dir is None or not os.path.exists(args.data_dir):
        print("Création d'un dataset factice pour la démonstration...")
        image_paths, labels = create_dummy_dataset(
            num_samples=200,
            image_size=tuple(args.image_size)
        )
        
        # Split train/val
        split_idx = int(len(image_paths) * 0.8)
        train_paths = image_paths[:split_idx]
        train_labels = labels[:split_idx]
        val_paths = image_paths[split_idx:]
        val_labels = labels[split_idx:]
        
        from .data_utils import MedicalImageDataset
        from torchvision import transforms
        
        train_transform = transforms.Compose([
            transforms.Resize(tuple(args.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(tuple(args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        train_dataset = MedicalImageDataset(
            train_paths, train_labels, train_transform, tuple(args.image_size)
        )
        val_dataset = MedicalImageDataset(
            val_paths, val_labels, val_transform, tuple(args.image_size)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        train_loader, val_loader = load_dataset_from_directory(
            args.data_dir,
            image_size=tuple(args.image_size)
        )
    
    # Créer le modèle
    model = GAMedicalClassifier(
        num_classes=2,
        multivector_dim=8,
        feature_dim=128,
        device=device
    ).to(device)
    
    # Compter les paramètres
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModèle créé avec {num_params:,} paramètres")
    
    # Entraînement
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()

