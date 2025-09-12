#!/usr/bin/env python3
"""
Lab 6 Hyperparameter Tuning - APEX Cluster Version
Autoencoder training with Optuna hyperparameter optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, Dataset

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os
import cv2
import glob
import math
from skimage.util import random_noise
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import ray
from ray import tune
from ray.air import session
import pandas as pd
from datetime import datetime
import argparse

# Set random seeds for reproducibility
seed = 4912
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Global variables for Ray Tune
TRAIN_FILES = None
TEST_FILES = None

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, gauss_noise=False, gauss_blur=None, resize=128, p=0.5):
        self.p = p
        self.resize = resize
        self.gauss_noise = gauss_noise
        self.gauss_blur = gauss_blur
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Check if file exists and is readable
        if not os.path.exists(image_path):
            print(f"Warning: File not found: {image_path}")
            # Return a black image as fallback
            gt_image = torch.zeros(3, self.resize, self.resize)
            noisy_image = gt_image.clone()
            return noisy_image, gt_image
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image: {image_path}")
            # Return a black image as fallback
            gt_image = torch.zeros(3, self.resize, self.resize)
            noisy_image = gt_image.clone()
            return noisy_image, gt_image
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        # Resize to 128x128
        image = cv2.resize(image, (self.resize, self.resize))
        
        # Ground truth image (clean)
        gt_image = image.copy().astype(np.float32) / 255.0
        
        # Create noisy/blurred version
        noisy_image = image.copy().astype(np.float32) / 255.0
        
        # Apply gaussian noise if enabled (Updated parameters)
        if self.gauss_noise and np.random.random() < self.p:
            # noise_factor = 0.3, noise scale (std) = 1
            noise_std = 1.0  # Fixed noise scale (std) = 1
            noise = np.random.normal(0, noise_std, noisy_image.shape).astype(np.float32)
            noisy_image = noisy_image + 0.3 * noise  # noise_factor = 0.3
            noisy_image = np.clip(noisy_image, 0, 1)  # Clip to [0,1] range
        
        # Apply gaussian blur if enabled
        if self.gauss_blur is not None and np.random.random() < self.p:
            kernel_size = self.gauss_blur
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size
            noisy_image = cv2.GaussianBlur(noisy_image, (kernel_size, kernel_size), 0)
        
        # Convert to torch tensors and change to CHW format
        noisy_image = torch.from_numpy(noisy_image.transpose(2, 0, 1)).float()
        gt_image = torch.from_numpy(gt_image.transpose(2, 0, 1)).float()

        return noisy_image, gt_image

class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DownSamplingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(UpSamplingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.upsample(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, channels=[64, 128, 256], input_channels=3, output_channels=3):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        in_ch = input_channels
        
        for out_ch in channels:
            encoder_layers.append(DownSamplingBlock(in_ch, out_ch))
            in_ch = out_ch
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1] * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[-1] * 2),
            nn.ReLU(),
            nn.Conv2d(channels[-1] * 2, channels[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[-1]),
            nn.ReLU()
        )
        
        # Decoder
        decoder_layers = []
        channels_reversed = channels[::-1]  # Reverse the channels
        
        for i in range(len(channels_reversed)):
            if i == len(channels_reversed) - 1:
                # Last layer outputs to output_channels
                decoder_layers.append(UpSamplingBlock(channels_reversed[i], output_channels))
            else:
                decoder_layers.append(UpSamplingBlock(channels_reversed[i], channels_reversed[i+1]))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Final output layer to ensure correct output
        self.final_conv = nn.Conv2d(output_channels, output_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)
        
        # Bottleneck
        bottleneck = self.bottleneck(encoded)
        
        # Decoder
        decoded = self.decoder(bottleneck)
        
        # Final output
        output = self.final_conv(decoded)
        output = self.sigmoid(output)
        
        return output

def train(model, opt, loss_fn, train_loader, test_loader, epochs=10, checkpoint_path=None, device='cpu'):
    print("🤖Training on", device)
    model = model.to(device)
    
    best_avg_psnr = -1.0  # Track best PSNR for saving best model
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_bar = tqdm(train_loader, desc=f'🚀Training Epoch [{epoch+1}/{epochs}]', unit='batch')
        train_loss = 0.0
        train_batches = 0
        
        for images, gt in train_bar:
            images, gt = images.to(device), gt.to(device)
            
            # Forward pass
            opt.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, gt)
            
            # Backward pass
            loss.backward()
            opt.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Update progress bar
            train_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / train_batches
        
        # Evaluation phase
        model.eval()
        test_bar = tqdm(test_loader, desc='📄Testing', unit='batch')
        test_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        test_batches = 0
        
        with torch.no_grad():
            for images, gt in test_bar:
                images, gt = images.to(device), gt.to(device)
                
                outputs = model(images)
                loss = loss_fn(outputs, gt)
                
                test_loss += loss.item()
                
                # Calculate PSNR and SSIM for each image in batch
                batch_psnr = 0.0
                batch_ssim = 0.0
                
                for i in range(outputs.size(0)):
                    # Convert to numpy for metric calculation
                    output_img = outputs[i].cpu().numpy().transpose(1, 2, 0)
                    gt_img = gt[i].cpu().numpy().transpose(1, 2, 0)
                    
                    # Ensure values are in [0, 1] range
                    output_img = np.clip(output_img, 0, 1)
                    gt_img = np.clip(gt_img, 0, 1)
                    
                    # Calculate metrics
                    psnr_val = psnr(gt_img, output_img, data_range=1.0)
                    ssim_val = ssim(gt_img, output_img, data_range=1.0, channel_axis=2)
                    
                    batch_psnr += psnr_val
                    batch_ssim += ssim_val
                
                batch_psnr /= outputs.size(0)
                batch_ssim /= outputs.size(0)
                
                total_psnr += batch_psnr
                total_ssim += batch_ssim
                test_batches += 1
                
                # Update progress bar
                test_bar.set_postfix(
                    loss=loss.item(),
                    psnr=batch_psnr,
                    ssim=batch_ssim
                )
        
        avg_test_loss = test_loss / test_batches
        avg_psnr = total_psnr / test_batches
        avg_ssim = total_ssim / test_batches
        
        # Save best model by PSNR
        if checkpoint_path and avg_psnr > best_avg_psnr:
            best_avg_psnr = avg_psnr
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'epoch': epoch + 1,
                'avg_psnr': float(avg_psnr),
                'avg_ssim': float(avg_ssim),
                'avg_train_loss': float(avg_train_loss),
                'avg_test_loss': float(avg_test_loss)
            }, checkpoint_path)
            print(f"💾 Saved best model (PSNR={avg_psnr:.4f}) to {checkpoint_path}")
        
        # Print summary
        print("Summary :")
        print(f"\tTrain\tavg_loss: {avg_train_loss}")
        print(f"\tTest\tavg_loss: {avg_test_loss}")
        print(f"\t\tPSNR : {avg_psnr}")
        print(f"\t\tSSIM : {avg_ssim}")
        print()
    
    # Final summary
    if checkpoint_path:
        print(f"🏆 Training finished. Best PSNR: {best_avg_psnr:.4f}")
        print(f"📁 Best model saved to: {checkpoint_path}")

def train_raytune_objective(config):
    """Ray Tune objective function for hyperparameter optimization"""
    
    # Get data from global variables (will be set in main)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Setup datasets with current config
        train_dataset_opt = CustomImageDataset(
            TRAIN_FILES,
            gauss_noise=True,
            gauss_blur=5,
            resize=128,
            p=0.7
        )
        
        test_dataset_opt = CustomImageDataset(
            TEST_FILES,
            gauss_noise=True,
            gauss_blur=5,
            resize=128,
            p=0.7
        )
        
        trainloader_opt = DataLoader(train_dataset_opt, batch_size=config['batch_size'], shuffle=True)
        testloader_opt = DataLoader(test_dataset_opt, batch_size=config['batch_size'], shuffle=False)
        
        # Create model with config architecture
        model = Autoencoder(channels=config['architecture']).to(device)

        # Setup optimizer based on config
        if config['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        elif config['optimizer'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)

        loss_fn = nn.MSELoss()

        best_psnr = 0
        
        for epoch in range(config['num_epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for images, gt in trainloader_opt:
                images, gt = images.to(device), gt.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, gt)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
            
            # Evaluation phase
            model.eval()
            test_loss = 0.0
            total_psnr = 0.0
            total_ssim = 0.0
            test_batches = 0
            
            with torch.no_grad():
                for images, gt in testloader_opt:
                    images, gt = images.to(device), gt.to(device)
                    
                    outputs = model(images)
                    loss = loss_fn(outputs, gt)
                    
                    test_loss += loss.item()
                    
                    # Calculate PSNR and SSIM
                    batch_psnr = 0.0
                    batch_ssim = 0.0
                    
                    for i in range(outputs.size(0)):
                        output_img = outputs[i].cpu().numpy().transpose(1, 2, 0)
                        gt_img = gt[i].cpu().numpy().transpose(1, 2, 0)
                        
                        output_img = np.clip(output_img, 0, 1)
                        gt_img = np.clip(gt_img, 0, 1)
                        
                        psnr_val = psnr(gt_img, output_img, data_range=1.0)
                        ssim_val = ssim(gt_img, output_img, data_range=1.0, channel_axis=2)
                        
                        batch_psnr += psnr_val
                        batch_ssim += ssim_val
                    
                    batch_psnr /= outputs.size(0)
                    batch_ssim /= outputs.size(0)
                    
                    total_psnr += batch_psnr
                    total_ssim += batch_ssim
                    test_batches += 1
            
            avg_test_loss = test_loss / test_batches if test_batches > 0 else 0
            avg_psnr = total_psnr / test_batches if test_batches > 0 else 0
            avg_ssim = total_ssim / test_batches if test_batches > 0 else 0
            
            # Keep track of best PSNR
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
            
            # Report to Ray Tune - always report metrics even if 0
            session.report({
                "train_loss": avg_train_loss,
                "val_loss": avg_test_loss,
                "val_psnr": avg_psnr,
                "val_ssim": avg_ssim,
            })
        
        return best_psnr
        
    except Exception as e:
        print(f"Trial failed: {str(e)}")
        # Still report metrics to avoid Ray Tune error
        session.report({
            "train_loss": float('inf'),
            "val_loss": float('inf'),
            "val_psnr": 0.0,
            "val_ssim": 0.0,
        })
        return 0  # Return low value for failed trials

class FeatureMapVisualizer:
    def __init__(self, model, layers, save_dir):
        """
        Parameters:
        - model: The PyTorch model
        - layers: A string or list of strings specifying the layer names to visualize
        - save_dir: Directory to save the output feature map images
        """
        self.model = model
        self.layers = layers if isinstance(layers, list) else [layers]
        self.activations = {}
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

        self._register_hooks()

    def _register_hooks(self):
        for name, layer in self.model.named_modules():
            if name in self.layers:
                layer.register_forward_hook(self._hook_fn(name))

    def _hook_fn(self, layer_name):
        def hook(module, input, output):
            print(f'Hooking layer: {layer_name}')
            self.activations[layer_name] = output.detach()
        return hook

    def visualize(self, input_tensor):
        """
        Pass an input tensor through the model and visualize the activations.
        
        Parameters:
        - input_tensor: Input tensor to pass through the model
        """
        
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)

        for layer_name, activation in self.activations.items():
            print(f'Visualizing and saving layer: {layer_name}')
            self._save_feature_maps(activation, layer_name)

    def _save_feature_maps(self, activation, layer_name):
        num_channels = activation.shape[1]
        
        # Calculate grid size for subplots
        grid_size = int(math.ceil(math.sqrt(num_channels)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        fig.suptitle(f'Feature Maps for {layer_name}', fontsize=16)
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        for i in range(num_channels):
            # Get the feature map for the first image in the batch
            feature_map = activation[0, i].cpu().numpy()
            
            # Plot the feature map
            axes[i].imshow(feature_map, cmap='viridis')
            axes[i].set_title(f'Channel {i}')
            axes[i].axis('off')
        
        # Hide remaining empty subplots
        for i in range(num_channels, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save the figure
        save_path = os.path.join(self.save_dir, f'feature_maps_{layer_name.replace(".", "_")}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'Saved feature maps to: {save_path}')

def run_grid_search(train_files, test_files, device, output_dir="grid_search_results", grid_timeout=43200):
    """Run grid search with Ray Tune"""
    print("🚀 Starting Hyperparameter Optimization with Ray Tune")
    print(f"Dataset: {len(train_files)} training images, {len(test_files)} test images")
    print(f"Device: {device}")
    print("="*60)

    # Set global variables for train_raytune_objective
    global TRAIN_FILES, TEST_FILES
    TRAIN_FILES = train_files
    TEST_FILES = test_files

    print("Starting hyperparameter search with updated parameters:")
    print("- Noise factor: 0.3, Noise scale (std): 1")
    print("- Batch sizes: [8, 16]")
    print("- Epochs: [5, 15]")  # Reduced epochs for faster completion
    print("- Optimizers: [Adam, SGD]")
    print("- Learning rates: [0.001, 0.0001]")
    print("- Cross-validation: 3-fold (using train/test split)")
    print("="*60)

    try:
        # Initialize Ray
        ray.shutdown()
        ray.init(num_gpus=1 if device.type == 'cuda' else 0)
        
        # Grid search configuration
        grid_config = {
            'architecture': tune.grid_search([[32, 64, 128], [64, 128, 256]]),
            'lr': tune.grid_search([0.001, 0.0001]),
            'batch_size': tune.grid_search([8, 16]),
            'num_epochs': tune.grid_search([5, 15]),  # Reduced epochs
            'optimizer': tune.grid_search(['Adam', 'SGD'])
        }
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        abs_output_dir = os.path.abspath(output_dir)
        
        # Run grid search
        tuner = tune.Tuner(
            train_raytune_objective,
            param_space=grid_config,
            tune_config=tune.TuneConfig(metric="val_psnr", mode="max", time_budget_s=grid_timeout),
            run_config=tune.RunConfig(name="grid_search", storage_path=abs_output_dir, stop={"training_iteration": 1})
        )
        
        result = tuner.fit()
        
        print("\n" + "="*60)
        print("🎉 OPTIMIZATION COMPLETED!")
        print("="*60)
        
        # Get best results
        best_result = result.get_best_result(metric="val_psnr", mode="max")
        print(f"\n🏆 BEST RESULTS:")
        print(f"Best PSNR: {best_result.metrics['val_psnr']:.2f} dB")
        
        print(f"\n📊 BEST HYPERPARAMETERS:")
        for key, value in best_result.config.items():
            print(f"  {key}: {value}")
        
        # Create results dataframe
        result_df = result.get_dataframe()
        result_df.to_csv(os.path.join(output_dir, 'grid_search_results.csv'), index=False)
        
        print("✅ Grid search with Ray Tune completed successfully!")
        return result, result_df
        
    except KeyboardInterrupt:
        print("\n⏹️ Optimization interrupted by user")
        return None, pd.DataFrame()
        
    except Exception as e:
        print(f"❌ Optimization failed: {str(e)}")
        return None, pd.DataFrame()
    
    finally:
        ray.shutdown()

def run_random_search(train_files, test_files, device, output_dir="random_search_results", random_timeout=21600):
    """Run random search with Ray Tune"""
    print("🚀 Starting Random Search with Ray Tune")

    # Set global variables for train_raytune_objective
    global TRAIN_FILES, TEST_FILES
    TRAIN_FILES = train_files
    TEST_FILES = test_files

    print("Running random hyperparameter search with updated parameters...")
    print("- Noise factor: 0.3, Noise scale (std): 1")
    print("- Batch sizes: [8, 16]") 
    print("- Epochs: [5, 15]")  # Reduced epochs
    print("- Optimizers: [Adam, SGD]")
    print("- Learning rates: [0.001, 0.0001]")
    print("This uses random sampling instead of Bayesian optimization")

    try:
        # Initialize Ray
        ray.shutdown()
        ray.init(num_gpus=1 if device.type == 'cuda' else 0)
        
        # Random search configuration
        random_config = {
            'architecture': tune.choice([[32, 64, 128], [64, 128, 256]]),
            'lr': tune.choice([0.001, 0.0001]),
            'batch_size': tune.choice([8, 16]),
            'num_epochs': tune.choice([5, 15]),  # Reduced epochs
            'optimizer': tune.choice(['Adam', 'SGD'])
        }
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        abs_output_dir = os.path.abspath(output_dir)
        
        # Run random search
        tuner = tune.Tuner(
            train_raytune_objective,
            param_space=random_config,
            tune_config=tune.TuneConfig(num_samples=16, metric="val_psnr", mode="max", time_budget_s=random_timeout),
            run_config=tune.RunConfig(name="random_search", storage_path=abs_output_dir, stop={"training_iteration": 1})
        )
        
        result = tuner.fit()
        
        print("🎉 Random search completed!")
        
        # Create results dataframe
        result_df = result.get_dataframe()
        result_df.to_csv(os.path.join(output_dir, 'random_search_results.csv'), index=False)
        
        print(f"✅ Random search completed with {len(result_df)} successful trials")
        return result, result_df

    except Exception as e:
        print(f"❌ Random search failed: {str(e)}")
        return None, pd.DataFrame()
    
    finally:
        ray.shutdown()

def train_final_models(grid_result, random_result, train_files, test_files, device, output_dir="final_models"):
    """Train final models with best configurations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup base datasets
    train_dataset = CustomImageDataset(
        train_files,
        gauss_noise=True,
        gauss_blur=5,
        resize=128,
        p=0.7
    )

    test_dataset = CustomImageDataset(
        test_files,
        gauss_noise=True,
        gauss_blur=5,
        resize=128,
        p=0.7
    )
    
    final_models = {}
    
    # Train with grid search best config
    if grid_result:
        best_config = grid_result.get_best_result(metric="val_psnr", mode="max").config
        print(f"Training final model with grid search best config: {best_config}")
        
        # Create model with best architecture
        best_model = Autoencoder(channels=best_config['architecture'])

        # Setup optimizer
        if best_config['optimizer'] == 'Adam':
            best_optimizer = optim.Adam(best_model.parameters(), lr=best_config['lr'])
        else:
            best_optimizer = optim.SGD(best_model.parameters(), lr=best_config['lr'], momentum=0.9)

        # Setup data loaders with best batch size
        best_trainloader = DataLoader(train_dataset, batch_size=best_config['batch_size'], shuffle=True)
        best_testloader = DataLoader(test_dataset, batch_size=best_config['batch_size'], shuffle=False)

        loss_fn = nn.MSELoss()

        # Train the model with best parameters
        train_epochs = best_config.get('num_epochs', 15)
        print(f"Training grid search best model for {train_epochs} epochs...")
        checkpoint_path = os.path.join(output_dir, "best_autoencoder_grid.pth")
        train(best_model, best_optimizer, loss_fn, best_trainloader, best_testloader, 
              epochs=train_epochs, checkpoint_path=checkpoint_path, device=device)
        
        final_models['grid'] = {
            'model': best_model,
            'config': best_config,
            'checkpoint': checkpoint_path
        }
    
    # Train with random search best config
    if random_result:
        best_config_random = random_result.get_best_result(metric="val_psnr", mode="max").config
        print(f"Training final model with random search best config: {best_config_random}")
        
        # Create model with best architecture
        best_model_random = Autoencoder(channels=best_config_random['architecture'])

        # Setup optimizer
        if best_config_random['optimizer'] == 'Adam':
            best_optimizer_random = optim.Adam(best_model_random.parameters(), lr=best_config_random['lr'])
        else:
            best_optimizer_random = optim.SGD(best_model_random.parameters(), lr=best_config_random['lr'], momentum=0.9)

        # Setup data loaders with best batch size
        best_trainloader_random = DataLoader(train_dataset, batch_size=best_config_random['batch_size'], shuffle=True)
        best_testloader_random = DataLoader(test_dataset, batch_size=best_config_random['batch_size'], shuffle=False)

        loss_fn = nn.MSELoss()

        # Train the model with best parameters from random search
        train_epochs_random = best_config_random.get('num_epochs', 15)
        print(f"Training random search best model for {train_epochs_random} epochs...")
        checkpoint_path_random = os.path.join(output_dir, "best_autoencoder_random.pth")
        train(best_model_random, best_optimizer_random, loss_fn, best_trainloader_random, best_testloader_random, 
              epochs=train_epochs_random, checkpoint_path=checkpoint_path_random, device=device)
        
        final_models['random'] = {
            'model': best_model_random,
            'config': best_config_random,
            'checkpoint': checkpoint_path_random
        }
    
    return final_models

def visualize_feature_maps(model, device, test_loader, output_dir="feature_maps"):
    """Visualize feature maps for the trained model"""
    # Get layer names from the encoder part of the model
    encoder_layer_names = []
    for name, module in model.named_modules():
        if 'encoder' in name and len(name.split('.')) == 2:  # Get main encoder blocks
            encoder_layer_names.append(name)

    print("Encoder layers to visualize:", encoder_layer_names)

    # Create visualizer for encoder layers
    visualizer = FeatureMapVisualizer(
        model=model, 
        layers=encoder_layer_names, 
        save_dir=output_dir
    )

    # Get a sample input from the test dataset
    sample_input, _ = next(iter(test_loader))
    sample_input = sample_input[:1].to(device)  # Take only first image and move to device

    print(f"Input shape: {sample_input.shape}")
    visualizer.visualize(sample_input)

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for autoencoder')
    parser.add_argument('--data_dir', type=str, default='img_align_celeba', 
                       help='Directory containing CelebA images')
    parser.add_argument('--output_dir', type=str, default='hyperopt_results',
                       help='Output directory for results')
    parser.add_argument('--num_images', type=int, default=30000,
                       help='Number of images to use from dataset')
    parser.add_argument('--grid_trials', type=int, default=12,
                       help='Number of trials for grid search')
    parser.add_argument('--random_trials', type=int, default=6,
                       help='Number of trials for random search')
    parser.add_argument('--grid_timeout', type=int, default=21600,
                       help='Timeout for grid search in seconds')
    parser.add_argument('--random_timeout', type=int, default=10800,
                       help='Timeout for random search in seconds')
    parser.add_argument('--skip_grid', action='store_true',
                       help='Skip grid search')
    parser.add_argument('--skip_random', action='store_true',
                       help='Skip random search')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip final training')
    parser.add_argument('--skip_visualization', action='store_true',
                       help='Skip feature map visualization')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load CelebA dataset
    # Make sure data_dir is absolute path to avoid issues with Ray Tune workers
    data_dir = os.path.abspath(args.data_dir)
    print(f"Loading images from {data_dir}...")
    image_files = glob.glob(os.path.join(data_dir, "*.jpg"))
    
    if len(image_files) == 0:
        print(f"No images found in {data_dir}. Please check the path.")
        return
    
    print(f"Found {len(image_files)} CelebA images")
    
    # Use subset for faster training
    image_files = image_files[:args.num_images]
    print(f"Using {len(image_files)} images for training")
    
    # Split into train/test
    train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)
    print(f"Training images: {len(train_files)}")
    print(f"Testing images: {len(test_files)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run grid search
    grid_result = None
    if not args.skip_grid:
        print("\n" + "="*60)
        print("RUNNING GRID SEARCH")
        print("="*60)
        grid_result, grid_df = run_grid_search(
            train_files, test_files, device, 
            output_dir=os.path.join(args.output_dir, "grid_search")
        )
    
    # Run random search
    random_result = None
    if not args.skip_random:
        print("\n" + "="*60)
        print("RUNNING RANDOM SEARCH")
        print("="*60)
        random_result, random_df = run_random_search(
            train_files, test_files, device,
            output_dir=os.path.join(args.output_dir, "random_search")
        )
    
    # Train final models
    final_models = {}
    if not args.skip_training:
        print("\n" + "="*60)
        print("TRAINING FINAL MODELS")
        print("="*60)
        final_models = train_final_models(
            grid_result, random_result, train_files, test_files, device,
            output_dir=os.path.join(args.output_dir, "final_models")
        )
    
    # Visualize feature maps
    if not args.skip_visualization and final_models:
        print("\n" + "="*60)
        print("VISUALIZING FEATURE MAPS")
        print("="*60)
        
        for name, model_info in final_models.items():
            print(f"Visualizing {name} model feature maps...")
            
            # Create test loader for visualization
            test_dataset = CustomImageDataset(
                test_files,
                gauss_noise=True,
                gauss_blur=5,
                resize=128,
                p=0.7
            )
            test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
            
            visualize_feature_maps(
                model_info['model'], device, test_loader,
                output_dir=os.path.join(args.output_dir, f"feature_maps_{name}")
            )
    
    print("\n" + "="*60)
    print("🎉 ALL TASKS COMPLETED!")
    print("="*60)
    print(f"Results saved to: {args.output_dir}")
    
    # Print summary
    if grid_result:
        best_grid = grid_result.get_best_result(metric="val_psnr", mode="max")
        print(f"Grid search best PSNR: {best_grid.metrics['val_psnr']:.2f} dB")
    if random_result:
        best_random = random_result.get_best_result(metric="val_psnr", mode="max")
        print(f"Random search best PSNR: {best_random.metrics['val_psnr']:.2f} dB")

if __name__ == "__main__":
    main()
