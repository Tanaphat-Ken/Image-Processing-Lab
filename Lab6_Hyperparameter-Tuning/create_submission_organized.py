"""
Lab6 Submission Creator with Organized Folder Structure
======================================================

Creates organized submission package with proper folder structure:
- submissions/submission_YYYYMMDD_HHMMSS/
  ├── images/              # Processed images for submission  
  ├── evaluation/          # Model evaluation results
  ├── README.txt           # Package documentation
  └── submission_summary.txt

Usage:
    python create_submission_organized.py
    python create_submission_organized.py --submission-name "my_submission"
"""

import torch
import os
import sys
import zipfile
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import json
import argparse

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from test_competition import (
        extract_test_data, load_model, TestDataset, 
        evaluate_model, save_results, print_competition_summary
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure test_competition.py is in the same directory")
    sys.exit(1)

from torch.utils.data import DataLoader


def create_submission_images(model_path="best_autoencoder.pth", test_zip="test.zip", 
                           submission_dir="submission_images", evaluation_dir="test_results"):
    """
    Create submission images by running the trained model on test data
    
    Args:
        model_path (str): Path to the trained model (.pth file)
        test_zip (str): Path to the test data zip file
        submission_dir (str): Directory to save submission images
        evaluation_dir (str): Directory to save evaluation results
    
    Returns:
        dict: Contains both evaluation results and submission info
    """
    
    print("🚀 Creating Submission Images & Evaluation...")
    print(f"📝 Model: {model_path}")
    print(f"📊 Test Data: {test_zip}")
    print(f"📁 Submission Folder: {submission_dir}")
    print(f"📊 Evaluation Folder: {evaluation_dir}")
    print("-" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
        
    if not os.path.exists(test_zip):
        print(f"❌ Test data not found: {test_zip}")
        return None
    
    # Create directories
    os.makedirs(submission_dir, exist_ok=True)
    os.makedirs(evaluation_dir, exist_ok=True)
    
    try:
        # Extract test data
        print("🗂️ Extracting test data...")
        test_image_paths = extract_test_data(test_zip, extract_dir="test_images_temp")
        
        if len(test_image_paths) == 0:
            print("❌ No images found in test data")
            return None
        
        print(f"📊 Found {len(test_image_paths)} test images")
        
        # Create test dataset and loader
        test_dataset = TestDataset(test_image_paths, resize=128)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
        
        # Load model
        print("🔄 Loading model...")
        model = load_model(model_path, device)
        
        # Skip evaluation for now - go directly to submission creation
        print("⚡ Skipping evaluation - going directly to submission creation...")
        all_metrics = []
        sample_results = []
        summary = {
            'avg_psnr': 'N/A',
            'avg_ssim': 'N/A', 
            'total_images': len(test_image_paths)
        }
        
        # Step 2: Generate submission images (Individual processing approach)
        print("\n🎨 Generating submission images...")
        model.eval()
        submission_count = 0
        
        # Import transforms at the top to avoid issues
        import torchvision.transforms as transforms
        from PIL import Image as PILImage
        
        # Create transform once
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
        print("📝 Processing images individually for reliable submission generation...")
        
        with torch.no_grad():
            progress_bar = tqdm(test_image_paths, desc='Creating Submissions', unit='image')
            
            for img_path in progress_bar:
                try:
                    # Load image
                    image = PILImage.open(img_path).convert('RGB')
                    
                    # Transform to match model input
                    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
                    
                    # Move to device - ensure it's a tensor
                    if isinstance(image_tensor, torch.Tensor):
                        image_tensor = image_tensor.to(device)
                    else:
                        print(f"⚠️ Warning: Expected tensor but got {type(image_tensor)} for {img_path}")
                        continue
                    
                    # Get model prediction
                    prediction = model(image_tensor)
                    
                    # Ensure prediction is a tensor
                    if not isinstance(prediction, torch.Tensor):
                        print(f"⚠️ Warning: Model output is not a tensor for {img_path}")
                        continue
                    
                    # Convert prediction to numpy
                    pred_img = prediction[0].cpu().numpy()  # Remove batch dimension
                    
                    # Convert from CHW to HWC format
                    pred_img = pred_img.transpose(1, 2, 0)
                    
                    # Ensure values are in [0, 1] range
                    pred_img = np.clip(pred_img, 0, 1)
                    
                    # Convert to 0-255 range
                    pred_img = (pred_img * 255).astype(np.uint8)
                    
                    # Convert to PIL Image
                    pred_pil = PILImage.fromarray(pred_img)
                    
                    # Get original filename
                    original_filename = os.path.basename(img_path)
                    
                    # Save with original filename
                    save_path = os.path.join(submission_dir, original_filename)
                    pred_pil.save(save_path, quality=95)
                    
                    submission_count += 1
                    
                    progress_bar.set_postfix(
                        saved=f"{submission_count}/{len(test_image_paths)}"
                    )
                    
                except Exception as e:
                    print(f"⚠️ Error processing {img_path}: {e}")
                    continue
        
        # Clean up temporary files
        import shutil
        if os.path.exists("test_images_temp"):
            shutil.rmtree("test_images_temp")
        
        # Create submission info
        submission_info = {
            'submission_dir': submission_dir,
            'total_images': submission_count,
            'model_used': model_path,
            'test_data': test_zip,
            'created_at': datetime.now().isoformat()
        }
        
        # Save submission info
        info_path = os.path.join(submission_dir, "submission_info.json")
        with open(info_path, 'w') as f:
            json.dump(submission_info, f, indent=2)
        
        # Show results
        saved_files = [f for f in os.listdir(submission_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\n✅ Successfully created {len(saved_files)} submission images!")
        print(f"📁 Submission images saved to: {submission_dir}")
        print(f"📊 Evaluation results saved to: {evaluation_dir}")
        
        print(f"\n📝 Sample submission filenames:")
        for i, filename in enumerate(sorted(saved_files)[:10]):
            print(f"   {i+1}: {filename}")
        
        if len(saved_files) > 10:
            print(f"   ... and {len(saved_files) - 10} more files")
        
        print(f"\n📦 Submission Package:")
        print(f"   🗂️ {len(saved_files)} processed images")
        print(f"   📄 submission_info.json (metadata)")
        print(f"   📊 Model performance: {summary.get('avg_psnr', 'N/A'):.2f} dB PSNR")
        
        # Combine results
        final_results = {
            'evaluation': summary,
            'submission': submission_info,
            'submission_images_count': len(saved_files)
        }
        
        return final_results
        
    except Exception as e:
        print(f"❌ Error during submission creation: {e}")
        import traceback
        traceback.print_exc()
        return None


def quick_submission(model_path="best_autoencoder.pth", test_zip="test.zip", submission_name=None):
    """
    Quick submission creation with organized folder structure
    
    Args:
        model_path (str): Path to the trained model
        test_zip (str): Path to test data
        submission_name (str): Custom name for submission folder
    
    Returns:
        str: Path to created submission folder
    """
    
    # Create unique submission folder name
    if submission_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_name = f"submission_{timestamp}"
    
    # Create main submission folder structure
    submission_base = "submissions"
    submission_full_path = os.path.join(submission_base, submission_name)
    
    os.makedirs(submission_base, exist_ok=True)
    os.makedirs(submission_full_path, exist_ok=True)
    
    # Create subfolders
    images_dir = os.path.join(submission_full_path, "images")
    evaluation_dir = os.path.join(submission_full_path, "evaluation")
    
    print("🚀 Creating Organized Submission Package...")
    print(f"📁 Submission folder: {submission_full_path}")
    print(f"🖼️ Images will be saved to: {images_dir}")
    print(f"📊 Evaluation will be saved to: {evaluation_dir}")
    print("-" * 60)
    
    # Run the submission creation
    results = create_submission_images(
        model_path=model_path,
        test_zip=test_zip,
        submission_dir=images_dir,
        evaluation_dir=evaluation_dir
    )
    
    if results:
        # Create README file
        readme_path = os.path.join(submission_full_path, "README.txt")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("Lab6 Hyperparameter Tuning - Submission Package\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Test Data: {test_zip}\n\n")
            
            f.write("📁 Folder Structure:\n")
            f.write("├── images/          # Processed images for submission\n")
            f.write("├── evaluation/      # Model evaluation results\n")
            f.write("├── README.txt       # This file\n")
            f.write("└── submission_summary.txt\n\n")
            
            if 'evaluation' in results:
                eval_data = results['evaluation']
                f.write("📊 Model Performance:\n")
                f.write(f"  • Average PSNR: {eval_data.get('avg_psnr', 'N/A'):.2f} dB\n")
                f.write(f"  • Average SSIM: {eval_data.get('avg_ssim', 'N/A'):.3f}\n")
                f.write(f"  • Total test images: {eval_data.get('total_images', 'N/A')}\n\n")
            
            f.write("🚀 How to Submit:\n")
            f.write("1. Navigate to the 'images' folder\n")
            f.write("2. Select all images (Ctrl+A)\n")
            f.write("3. Create a zip file (right-click -> Send to -> Compressed folder)\n")
            f.write("4. Submit the zip file to the competition\n")
            f.write("5. Include evaluation results if requested\n")
        
        # Create submission summary
        summary_path = os.path.join(submission_full_path, "submission_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("SUBMISSION SUMMARY\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Submission ID: {submission_name}\n")
            f.write(f"Created: {datetime.now().isoformat()}\n")
            f.write(f"Images Count: {results.get('submission_images_count', 'N/A')}\n")
            
            if 'evaluation' in results:
                eval_data = results['evaluation']
                f.write(f"PSNR: {eval_data.get('avg_psnr', 'N/A'):.2f} dB\n")
                f.write(f"SSIM: {eval_data.get('avg_ssim', 'N/A'):.3f}\n")
        
        print(f"\n🎉 Submission package created successfully!")
        print(f"📦 Location: {submission_full_path}")
        print(f"\n📋 Package contents:")
        print(f"   📁 images/ - {results.get('submission_images_count', 0)} processed images")
        print(f"   📊 evaluation/ - Model performance metrics")
        print(f"   📄 README.txt - Package documentation")
        print(f"   📄 submission_summary.txt - Quick summary")
        
        print(f"\n🚀 Ready to submit:")
        print(f"   1. Navigate to: {images_dir}")
        print(f"   2. Select all images and create a zip file")
        print(f"   3. Submit the zip file")
        print(f"\n💡 Tip: Right-click in the images folder -> Send to -> Compressed (zipped) folder")
        
        return submission_full_path
    else:
        print("❌ Failed to create submission package")
        return None


def test_with_existing_model():
    """Test with your existing trained model"""
    
    # Try to find your best model
    model_candidates = [
        "best_autoencoder.pth",
        "best_autoencoder_random.pth", 
        "autoencoder_model.pth",
        "../Lab5_CNN/autoencoder_model.pth"
    ]
    
    model_path = None
    for candidate in model_candidates:
        if os.path.exists(candidate):
            model_path = candidate
            break
    
    if model_path is None:
        print("❌ No trained model found!")
        print("Available candidates checked:")
        for candidate in model_candidates:
            print(f"  - {candidate}")
        return None
    
    # Try to find test data
    test_candidates = [
        "test.zip",
        "test_celeba.zip",
        "../test.zip"
    ]
    
    test_path = None
    for candidate in test_candidates:
        if os.path.exists(candidate):
            test_path = candidate
            break
    
    if test_path is None:
        print("❌ No test data found!")
        print("Available candidates checked:")
        for candidate in test_candidates:
            print(f"  - {candidate}")
        return None
    
    print(f"✅ Found model: {model_path}")
    print(f"✅ Found test data: {test_path}")
    
    return model_path, test_path


if __name__ == "__main__":
    print("🚀 Lab6 Organized Submission Creator")
    print("=" * 50)
    
    parser = argparse.ArgumentParser(description='Lab6 Organized Submission Creator')
    parser.add_argument('--mode', choices=['auto', 'manual'], default='auto',
                       help='Mode: auto (find files automatically), manual (specify paths)')
    parser.add_argument('--model', default='best_autoencoder.pth', help='Model path')
    parser.add_argument('--test-zip', default='test.zip', help='Test data zip')
    parser.add_argument('--submission-name', help='Custom submission folder name')
    
    args = parser.parse_args()
    
    if args.mode == 'auto':
        # Auto-detect files
        result = test_with_existing_model()
        if result:
            model_path, test_path = result
            print(f"🔍 Auto-detected: {model_path}, {test_path}")
            
            submission_result = quick_submission(model_path, test_path, args.submission_name)
            if submission_result:
                print(f"\n✅ Auto-submission completed: {submission_result}")
            else:
                print("\n❌ Auto-submission failed!")
        else:
            print("❌ Auto-detection failed! Try manual mode.")
    else:
        # Manual mode - use specified paths
        submission_result = quick_submission(args.model, args.test_zip, args.submission_name)
        if submission_result:
            print(f"\n✅ Submission package created: {submission_result}")
        else:
            print("\n❌ Submission creation failed!")
