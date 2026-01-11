"""
Conditional GAN (cGAN) for generating synthetic COVID-19 chest X-ray images.

Based on the approach from:
"Machine-Learning-Based COVID-19 Detection with Enhanced cGAN Technique Using X-ray Images"
(Electronics 2022)

This cGAN generates synthetic COVID-19 images to balance the class distribution
in the training dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from typing import Tuple, Optional
from PIL import Image
import torchvision.transforms as transforms


class Generator(nn.Module):
    """
    Generator network for conditional GAN.
    Takes noise vector + class label and generates synthetic chest X-ray images.
    """
    
    def __init__(self, latent_dim: int = 100, num_classes: int = 2, img_size: int = 224):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Embedding for class label
        self.label_embedding = nn.Embedding(num_classes, 50)
        
        # Input: noise (latent_dim) + label embedding (50) = latent_dim + 50
        self.fc = nn.Linear(latent_dim + 50, 7 * 7 * 256)
        
        # Upsampling layers
        self.conv_blocks = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 28x28 -> 56x56
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 56x56 -> 112x112
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 112x112 -> 224x224
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noise: Random noise tensor (B, latent_dim)
            labels: Class labels (B,)
        
        Returns:
            Generated images (B, 1, img_size, img_size)
        """
        # Embed labels
        label_emb = self.label_embedding(labels)  # (B, 50)
        
        # Concatenate noise and label embedding
        input_vec = torch.cat([noise, label_emb], dim=1)  # (B, latent_dim + 50)
        
        # Fully connected layer
        x = self.fc(input_vec)  # (B, 7*7*256)
        x = x.view(x.size(0), 256, 7, 7)  # (B, 256, 7, 7)
        
        # Upsampling
        x = self.conv_blocks(x)  # (B, 1, 224, 224)
        
        return x


class Discriminator(nn.Module):
    """
    Discriminator network for conditional GAN.
    Takes image + class label and predicts if image is real or fake.
    """
    
    def __init__(self, num_classes: int = 2, img_size: int = 224):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Embedding for class label
        self.label_embedding = nn.Embedding(num_classes, img_size * img_size)
        
        # Convolutional layers
        self.conv_blocks = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 14x14 -> 7x7
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7 + img_size * img_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Image tensor (B, 1, img_size, img_size)
            labels: Class labels (B,)
        
        Returns:
            Probability that image is real (B, 1)
        """
        # Convolutional features
        img_features = self.conv_blocks(images)  # (B, 512, 7, 7)
        img_features = img_features.view(img_features.size(0), -1)  # (B, 512*7*7)
        
        # Label embedding
        label_emb = self.label_embedding(labels)  # (B, img_size*img_size)
        
        # Concatenate
        combined = torch.cat([img_features, label_emb], dim=1)  # (B, 512*7*7 + img_size*img_size)
        
        # Output
        output = self.fc(combined)  # (B, 1)
        
        return output


class ConditionalGAN:
    """
    Conditional GAN for generating synthetic COVID-19 chest X-ray images.
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 2,
        img_size: int = 224,
        device: str = 'cpu'
    ):
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.device = device
        
        # Initialize networks
        self.generator = Generator(latent_dim, num_classes, img_size).to(device)
        self.discriminator = Discriminator(num_classes, img_size).to(device)
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Optimizers
        self.g_optimizer = None
        self.d_optimizer = None
    
    def setup_optimizers(self, lr_g: float = 0.0002, lr_d: float = 0.0002, beta1: float = 0.5):
        """Setup optimizers for generator and discriminator."""
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=lr_g, betas=(beta1, 0.999)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d, betas=(beta1, 0.999)
        )
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        target_class: int = 1,  # Class to generate (1 = COVID-19)
        num_d_steps: int = 1
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader with real images
            target_class: Class label to generate
            num_d_steps: Number of discriminator steps per generator step
        
        Returns:
            Tuple of (generator_loss, discriminator_loss)
        """
        self.generator.train()
        self.discriminator.train()
        
        g_losses = []
        d_losses = []
        
        for batch_idx, (real_images, real_labels) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)
            real_labels = real_labels.to(self.device)
            
            # Create labels for real and fake
            real_labels_tensor = torch.ones(batch_size, 1, device=self.device) * 0.9  # Label smoothing
            fake_labels_tensor = torch.zeros(batch_size, 1, device=self.device)
            
            # =====================
            # Train Discriminator
            # =====================
            self.d_optimizer.zero_grad()
            
            # Real images
            real_pred = self.discriminator(real_images, real_labels)
            d_loss_real = self.criterion(real_pred, real_labels_tensor)
            
            # Fake images
            noise = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_labels = torch.full((batch_size,), target_class, dtype=torch.long, device=self.device)
            fake_images = self.generator(noise, fake_labels)
            fake_pred = self.discriminator(fake_images.detach(), fake_labels)
            d_loss_fake = self.criterion(fake_pred, fake_labels_tensor)
            
            # Total discriminator loss
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            self.d_optimizer.step()
            
            d_losses.append(d_loss.item())
            
            # =====================
            # Train Generator
            # =====================
            if batch_idx % num_d_steps == 0:
                self.g_optimizer.zero_grad()
                
                # Generate fake images
                noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_labels = torch.full((batch_size,), target_class, dtype=torch.long, device=self.device)
                fake_images = self.generator(noise, fake_labels)
                
                # Try to fool discriminator
                fake_pred = self.discriminator(fake_images, fake_labels)
                g_loss = self.criterion(fake_pred, real_labels_tensor)
                
                g_loss.backward()
                self.g_optimizer.step()
                
                g_losses.append(g_loss.item())
        
        avg_g_loss = np.mean(g_losses) if g_losses else 0.0
        avg_d_loss = np.mean(d_losses) if d_losses else 0.0
        
        return avg_g_loss, avg_d_loss
    
    def generate_images(
        self,
        num_images: int,
        target_class: int = 1,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Generate synthetic images.
        
        Args:
            num_images: Number of images to generate
            target_class: Class label to generate
            batch_size: Batch size for generation
        
        Returns:
            Generated images tensor (num_images, 1, img_size, img_size)
        """
        self.generator.eval()
        
        all_images = []
        
        with torch.no_grad():
            for i in range(0, num_images, batch_size):
                current_batch_size = min(batch_size, num_images - i)
                noise = torch.randn(current_batch_size, self.latent_dim, device=self.device)
                labels = torch.full((current_batch_size,), target_class, dtype=torch.long, device=self.device)
                
                fake_images = self.generator(noise, labels)
                all_images.append(fake_images.cpu())
        
        return torch.cat(all_images, dim=0)
    
    def save_generator(self, path: str):
        """Save generator model."""
        torch.save(self.generator.state_dict(), path)
    
    def load_generator(self, path: str):
        """Load generator model."""
        self.generator.load_state_dict(torch.load(path, map_location=self.device))
    
    def save_synthetic_images(
        self,
        num_images: int,
        output_dir: str,
        target_class: int = 1,
        prefix: str = "synthetic_covid"
    ):
        """
        Generate and save synthetic images to disk.
        
        Args:
            num_images: Number of images to generate
            output_dir: Directory to save images
            target_class: Class label to generate
            prefix: Filename prefix
        """
        os.makedirs(output_dir, exist_ok=True)
        
        images = self.generate_images(num_images, target_class)
        
        # Transform from [-1, 1] to [0, 255]
        images = (images + 1) / 2  # [0, 1]
        images = torch.clamp(images, 0, 1)
        
        for i, img_tensor in enumerate(images):
            # Convert to numpy
            img_np = img_tensor.squeeze().numpy()  # (224, 224)
            img_np = (img_np * 255).astype(np.uint8)
            
            # Save as PNG
            img = Image.fromarray(img_np, mode='L')
            filename = os.path.join(output_dir, f"{prefix}_{i:05d}.png")
            img.save(filename)
        
        print(f"Saved {num_images} synthetic images to {output_dir}")

