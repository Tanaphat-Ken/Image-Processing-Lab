"""
Lab6 Hyperparameter Tuning - Competition Test Script
====================================================

This script evaluates the trained autoencoder model on the hidden test dataset
according to the competition evaluation rules.

Evaluation Metrics:
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR) 
- Structural Similarity Index (SSIM)

Usage:
    python test_competition.py --model_path best_autoencoder.pth --test_data test.zip
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image
import os
import glob
import zipfile
import argparse
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import json
from datetime import datetime

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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

class TestDataset(Dataset):
    """Dataset for loading test images with corruptions"""
    def __init__(self, image_paths, resize=128):
        self.image_paths = image_paths
        self.resize = resize
        
        # Base transforms for consistency
        self.base_transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (self.resize, self.resize), (0, 0, 0))
        
        # Apply transform
        image_tensor = self.base_transform(image)
        
        return image_tensor, os.path.basename(img_path)

def extract_test_data(test_zip_path, extract_dir="test_images"):
    """Extract test data from zip file"""
    print(f"🗂️ Extracting test data from {test_zip_path}...")
    
    # Create extraction directory
    os.makedirs(extract_dir, exist_ok=True)
    
    # Extract zip file
    with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(extract_dir, '**', ext), recursive=True))
    
    print(f"📊 Found {len(image_files)} test images")
    return image_files

def load_model(model_path, device):
    """Load trained model"""
    print(f"🔄 Loading model from {model_path}...")
    
    # Try to load model with different architectures
    architectures = [
        [64, 128, 256],    # Most common
        [32, 64, 128],     # Smaller
        [64, 128, 256, 512] # Larger
    ]
    
    model = None
    for arch in architectures:
        try:
            model = Autoencoder(channels=arch)
            state_dict = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
                print(f"✅ Loaded model with architecture {arch} (checkpoint format)")
            else:
                model.load_state_dict(state_dict)
                print(f"✅ Loaded model with architecture {arch} (state dict format)")
            
            model = model.to(device)
            model.eval()
            break
            
        except Exception as e:
            print(f"❌ Failed to load with architecture {arch}: {e}")
            continue
    
    if model is None:
        raise ValueError("Could not load model with any architecture. Please check the model file.")
    
    return model

def calculate_metrics(predicted, target):
    """Calculate MSE, PSNR, and SSIM metrics"""
    # Convert tensors to numpy arrays
    if torch.is_tensor(predicted):
        predicted = predicted.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    # Ensure values are in [0, 1] range
    predicted = np.clip(predicted, 0, 1)
    target = np.clip(target, 0, 1)
    
    # Calculate MSE
    mse = np.mean((predicted - target) ** 2)
    
    # Calculate PSNR
    if predicted.ndim == 3:  # Single image CHW format
        predicted_hw = predicted.transpose(1, 2, 0)
        target_hw = target.transpose(1, 2, 0)
    else:  # Already HW or HWC format
        predicted_hw = predicted
        target_hw = target
    
    try:
        psnr_val = psnr(target_hw, predicted_hw, data_range=1.0)
    except:
        psnr_val = 0
    
    # Calculate SSIM
    try:
        if predicted_hw.ndim == 3 and predicted_hw.shape[2] == 3:  # Color image
            ssim_val = ssim(target_hw, predicted_hw, data_range=1.0, channel_axis=2)
        else:  # Grayscale
            ssim_val = ssim(target_hw, predicted_hw, data_range=1.0)
    except:
        ssim_val = 0
    
    return mse, psnr_val, ssim_val

def add_test_corruptions(image_tensor):
    """Add various corruptions to simulate test conditions"""
    corrupted = image_tensor.clone()
    
    # Random corruption type
    corruption_type = np.random.randint(0, 4)
    
    if corruption_type == 0:
        # Gaussian noise
        noise = torch.randn_like(corrupted) * 0.1
        corrupted = torch.clamp(corrupted + noise, 0, 1)
    
    elif corruption_type == 1:
        # Salt and pepper noise
        mask = torch.rand_like(corrupted)
        salt_mask = mask > 0.95
        pepper_mask = mask < 0.05
        corrupted[salt_mask] = 1
        corrupted[pepper_mask] = 0
    
    elif corruption_type == 2:
        # Random patches masking
        c, h, w = corrupted.shape
        patch_size = np.random.randint(10, 30)
        x = np.random.randint(0, max(1, w - patch_size))
        y = np.random.randint(0, max(1, h - patch_size))
        corrupted[:, y:y+patch_size, x:x+patch_size] = 0
    
    else:
        # Gaussian blur
        # Simple blur simulation by downsampling and upsampling
        c, h, w = corrupted.shape
        downscale = np.random.randint(2, 5)
        small_h, small_w = max(1, h // downscale), max(1, w // downscale)
        
        corrupted_resized = torch.nn.functional.interpolate(
            corrupted.unsqueeze(0), 
            size=(small_h, small_w), 
            mode='bilinear', 
            align_corners=False
        )
        corrupted = torch.nn.functional.interpolate(
            corrupted_resized, 
            size=(h, w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
    
    return corrupted

def evaluate_model(model, test_loader, device, save_samples=True):
    """Evaluate model on test dataset"""
    print("🧪 Evaluating model on test dataset...")
    
    model.eval()
    all_metrics = []
    sample_results = []
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Testing', unit='batch')
        
        for batch_idx, (images, filenames) in enumerate(test_bar):
            images = images.to(device)
            
            # Add test corruptions to simulate real test conditions
            corrupted_images = torch.stack([add_test_corruptions(img) for img in images])
            corrupted_images = corrupted_images.to(device)
            
            # Get model predictions
            predictions = model(corrupted_images)
            
            # Calculate metrics for each image in batch
            batch_mse = []
            batch_psnr = []
            batch_ssim = []
            
            for i in range(images.size(0)):
                # Calculate metrics comparing prediction to original (clean) image
                mse, psnr_val, ssim_val = calculate_metrics(
                    predictions[i], images[i]
                )
                
                batch_mse.append(mse)
                batch_psnr.append(psnr_val)
                batch_ssim.append(ssim_val)
                
                # Store individual results
                all_metrics.append({
                    'filename': filenames[i],
                    'mse': float(mse),
                    'psnr': float(psnr_val),
                    'ssim': float(ssim_val)
                })
                
                # Save sample results for the first few images
                if save_samples and len(sample_results) < 10:
                    sample_results.append({
                        'filename': filenames[i],
                        'original': images[i].cpu().numpy(),
                        'corrupted': corrupted_images[i].cpu().numpy(),
                        'prediction': predictions[i].cpu().numpy(),
                        'mse': float(mse),
                        'psnr': float(psnr_val),
                        'ssim': float(ssim_val)
                    })
            
            # Update progress bar
            avg_mse = np.mean(batch_mse)
            avg_psnr = np.mean(batch_psnr)
            avg_ssim = np.mean(batch_ssim)
            
            test_bar.set_postfix(
                mse=avg_mse,
                psnr=avg_psnr,
                ssim=avg_ssim
            )
    
    return all_metrics, sample_results

def save_results(all_metrics, sample_results, output_dir="test_results"):
    """Save evaluation results"""
    print(f"💾 Saving results to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate summary statistics
    mse_values = [m['mse'] for m in all_metrics]
    psnr_values = [m['psnr'] for m in all_metrics]
    ssim_values = [m['ssim'] for m in all_metrics]
    
    summary = {
        'total_images': len(all_metrics),
        'avg_mse': float(np.mean(mse_values)),
        'std_mse': float(np.std(mse_values)),
        'avg_psnr': float(np.mean(psnr_values)),
        'std_psnr': float(np.std(psnr_values)),
        'avg_ssim': float(np.mean(ssim_values)),
        'std_ssim': float(np.std(ssim_values)),
        'min_psnr': float(np.min(psnr_values)),
        'max_psnr': float(np.max(psnr_values)),
        'median_psnr': float(np.median(psnr_values)),
        'min_ssim': float(np.min(ssim_values)),
        'max_ssim': float(np.max(ssim_values)),
        'median_ssim': float(np.median(ssim_values)),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save detailed results to CSV
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
    
    # Save summary to JSON
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save summary for competition submission
    competition_summary = {
        'team_name': 'YourTeamName',  # Update this
        'model_name': 'Autoencoder_Hyperparameter_Tuned',
        'total_test_images': summary['total_images'],
        'average_mse': float(summary['avg_mse']),
        'average_psnr': float(summary['avg_psnr']),
        'average_ssim': float(summary['avg_ssim']),
        'submission_time': summary['timestamp']
    }
    
    with open(os.path.join(output_dir, 'competition_submission.json'), 'w') as f:
        json.dump(competition_summary, f, indent=2)
    
    # Create visualizations
    create_visualizations(all_metrics, sample_results, output_dir)
    
    return summary

def create_visualizations(all_metrics, sample_results, output_dir):
    """Create visualization plots"""
    
    # Extract metric values
    mse_values = [m['mse'] for m in all_metrics]
    psnr_values = [m['psnr'] for m in all_metrics]
    ssim_values = [m['ssim'] for m in all_metrics]
    
    # Create metric distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # MSE histogram
    axes[0, 0].hist(mse_values, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].axvline(np.mean(mse_values), color='darkred', linestyle='--', 
                       label=f'Mean: {np.mean(mse_values):.6f}')
    axes[0, 0].set_xlabel('MSE')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('MSE Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PSNR histogram
    axes[0, 1].hist(psnr_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].axvline(np.mean(psnr_values), color='darkblue', linestyle='--', 
                       label=f'Mean: {np.mean(psnr_values):.2f} dB')
    axes[0, 1].set_xlabel('PSNR (dB)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('PSNR Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # SSIM histogram
    axes[1, 0].hist(ssim_values, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].axvline(np.mean(ssim_values), color='darkgreen', linestyle='--', 
                       label=f'Mean: {np.mean(ssim_values):.3f}')
    axes[1, 0].set_xlabel('SSIM')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('SSIM Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter plot: PSNR vs SSIM
    axes[1, 1].scatter(psnr_values, ssim_values, alpha=0.6, s=20)
    axes[1, 1].set_xlabel('PSNR (dB)')
    axes[1, 1].set_ylabel('SSIM')
    axes[1, 1].set_title('PSNR vs SSIM Correlation')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create sample results visualization
    if sample_results:
        create_sample_visualization(sample_results, output_dir)

def create_sample_visualization(sample_results, output_dir):
    """Create visualization of sample results"""
    
    n_samples = min(6, len(sample_results))
    fig, axes = plt.subplots(3, n_samples, figsize=(4*n_samples, 12))
    
    if n_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_samples):
        sample = sample_results[i]
        
        # Convert from CHW to HWC for display
        original = sample['original'].transpose(1, 2, 0)
        corrupted = sample['corrupted'].transpose(1, 2, 0)
        prediction = sample['prediction'].transpose(1, 2, 0)
        
        # Ensure values are in [0, 1]
        original = np.clip(original, 0, 1)
        corrupted = np.clip(corrupted, 0, 1)
        prediction = np.clip(prediction, 0, 1)
        
        # Display images
        axes[0, i].imshow(original)
        axes[0, i].set_title(f'Original\n{sample["filename"]}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(corrupted)
        axes[1, i].set_title('Corrupted Input')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(prediction)
        axes[2, i].set_title(f'Prediction\nPSNR: {sample["psnr"]:.2f}\nSSIM: {sample["ssim"]:.3f}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_results.png'), dpi=300, bbox_inches='tight')
    plt.close()

def print_competition_summary(summary):
    """Print formatted competition summary"""
    print("\n" + "="*80)
    print("🏆 COMPETITION EVALUATION RESULTS")
    print("="*80)
    print(f"📊 Total Test Images: {summary['total_images']}")
    print(f"📈 Average MSE: {summary['avg_mse']:.6f} ± {summary['std_mse']:.6f}")
    print(f"📈 Average PSNR: {summary['avg_psnr']:.4f} ± {summary['std_psnr']:.4f} dB")
    print(f"📈 Average SSIM: {summary['avg_ssim']:.4f} ± {summary['std_ssim']:.4f}")
    print(f"📊 PSNR Range: {summary['min_psnr']:.2f} - {summary['max_psnr']:.2f} dB")
    print(f"📊 SSIM Range: {summary['min_ssim']:.3f} - {summary['max_ssim']:.3f}")
    print(f"📊 Median PSNR: {summary['median_psnr']:.2f} dB")
    print(f"📊 Median SSIM: {summary['median_ssim']:.3f}")
    print("="*80)
    print("📁 Results saved to:")
    print("   - detailed_results.csv: Per-image results")
    print("   - summary.json: Overall statistics")
    print("   - competition_submission.json: Competition format")
    print("   - metrics_distribution.png: Metric distributions")
    print("   - sample_results.png: Sample predictions")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Test autoencoder model for competition')
    parser.add_argument('--model_path', type=str, default='best_autoencoder.pth',
                       help='Path to trained model file')
    parser.add_argument('--test_data', type=str, default='test.zip',
                       help='Path to test data zip file')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for testing')
    parser.add_argument('--output_dir', type=str, default='test_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Extract test data
    test_image_paths = extract_test_data(args.test_data)
    
    # Create test dataset and loader
    test_dataset = TestDataset(test_image_paths)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Evaluate model
    all_metrics, sample_results = evaluate_model(model, test_loader, device)
    
    # Save results
    summary = save_results(all_metrics, sample_results, args.output_dir)
    
    # Print summary
    print_competition_summary(summary)

if __name__ == "__main__":
    main()
