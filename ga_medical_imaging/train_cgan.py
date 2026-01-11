#!/usr/bin/env python3
"""
Train a conditional GAN (cGAN) to generate synthetic COVID-19 chest X-ray images.

Based on: "Machine-Learning-Based COVID-19 Detection with Enhanced cGAN Technique Using X-ray Images"
(Electronics 2022)

This script trains a cGAN to generate synthetic COVID-19 images to balance the class distribution.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import numpy as np

from ga_medical_imaging.cgan_generator import ConditionalGAN
from ga_medical_imaging.data_utils import load_dataset_from_directory, MedicalImageDataset
from torchvision import transforms


def train_cgan(
    cgan: ConditionalGAN,
    train_loader: DataLoader,
    num_epochs: int = 100,
    target_class: int = 1,
    save_dir: str = 'checkpoints/cgan',
    save_every: int = 10
):
    """
    Train the conditional GAN.
    
    Args:
        cgan: ConditionalGAN instance
        train_loader: DataLoader with real images
        num_epochs: Number of training epochs
        target_class: Class to generate (1 = COVID-19)
        save_dir: Directory to save checkpoints
        save_every: Save checkpoint every N epochs
    """
    os.makedirs(save_dir, exist_ok=True)
    
    cgan.setup_optimizers()
    
    print(f"\n{'='*70}")
    print(f"TRAINING CONDITIONAL GAN")
    print(f"{'='*70}")
    print(f"Target class: {target_class} (COVID-19)")
    print(f"Number of epochs: {num_epochs}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"{'='*70}\n")
    
    g_losses = []
    d_losses = []
    
    for epoch in range(1, num_epochs + 1):
        g_loss, d_loss = cgan.train_epoch(train_loader, target_class=target_class)
        
        g_losses.append(g_loss)
        d_losses.append(d_loss)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{num_epochs}:")
            print(f"  Generator Loss: {g_loss:.4f}")
            print(f"  Discriminator Loss: {d_loss:.4f}")
        
        # Save checkpoint
        if epoch % save_every == 0:
            checkpoint_path = os.path.join(save_dir, f'cgan_epoch_{epoch}.pth')
            cgan.save_generator(checkpoint_path)
            print(f"  ✓ Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(save_dir, 'cgan_final.pth')
    cgan.save_generator(final_path)
    print(f"\n✓ Training complete! Final model saved to: {final_path}")
    
    return g_losses, d_losses


def main():
    parser = argparse.ArgumentParser(
        description='Train conditional GAN for synthetic COVID-19 image generation'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--latent_dim', type=int, default=100,
                       help='Latent dimension for generator')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size')
    parser.add_argument('--target_class', type=int, default=1,
                       help='Class to generate (1=COVID-19, 0=No Finding)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu, cuda, or auto)')
    parser.add_argument('--save_dir', type=str, default='checkpoints/cgan',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"\nLoading dataset from: {args.data_dir}")
    train_loader, _ = load_dataset_from_directory(
        args.data_dir,
        image_size=(args.img_size, args.img_size)
    )
    
    # Filter for target class only (for training cGAN)
    # We'll train on all images but generate only target class
    print(f"Using all {len(train_loader.dataset)} training images for cGAN training")
    
    # Create cGAN
    cgan = ConditionalGAN(
        latent_dim=args.latent_dim,
        num_classes=2,
        img_size=args.img_size,
        device=device
    )
    
    # Train
    g_losses, d_losses = train_cgan(
        cgan,
        train_loader,
        num_epochs=args.num_epochs,
        target_class=args.target_class,
        save_dir=args.save_dir,
        save_every=args.save_every
    )
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Final Generator Loss: {g_losses[-1]:.4f}")
    print(f"Final Discriminator Loss: {d_losses[-1]:.4f}")
    print("="*70)


if __name__ == '__main__':
    main()

