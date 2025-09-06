"""
Quick Test Runner for Lab6 Competition Evaluation
===============================================

This script can be run directly from the notebook or command line
to evaluate your trained autoencoder model.
"""

import torch
import os
import sys
import zipfile
import glob
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from test_competition import (
        extract_test_data, load_model, TestDataset, 
        evaluate_model, save_results, print_competition_summary,
        create_visualizations
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure test_competition.py is in the same directory")
    sys.exit(1)

from torch.utils.data import DataLoader

def quick_test(model_path="best_autoencoder.pth", test_zip="test.zip", output_dir="test_results"):
    """
    Quick test function that can be called from notebook or script
    
    Args:
        model_path (str): Path to the trained model (.pth file)
        test_zip (str): Path to the test data zip file
        output_dir (str): Directory to save results
    
    Returns:
        dict: Summary of evaluation results
    """
    
    print("🚀 Starting Competition Evaluation...")
    print(f"📝 Model: {model_path}")
    print(f"📊 Test Data: {test_zip}")
    print(f"💾 Output: {output_dir}")
    print("-" * 50)
    
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
    
    try:
        # Extract test data
        test_image_paths = extract_test_data(test_zip, extract_dir="test_images_temp")
        
        if len(test_image_paths) == 0:
            print("❌ No images found in test data")
            return None
        
        # Create test dataset and loader
        test_dataset = TestDataset(test_image_paths, resize=128)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
        
        # Load model
        model = load_model(model_path, device)
        
        # Evaluate model
        all_metrics, sample_results = evaluate_model(model, test_loader, device, save_samples=True)
        
        # Save results
        summary = save_results(all_metrics, sample_results, output_dir)
        
        # Print summary
        print_competition_summary(summary)
        
        # Clean up temporary files
        import shutil
        if os.path.exists("test_images_temp"):
            shutil.rmtree("test_images_temp")
        
        return summary
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
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
        print("❌ Could not find trained model. Available files:")
        for f in os.listdir("."):
            if f.endswith(".pth"):
                print(f"   - {f}")
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
        print("❌ Could not find test data. Available zip files:")
        for f in os.listdir("."):
            if f.endswith(".zip"):
                print(f"   - {f}")
        return None
    
    print(f"✅ Found model: {model_path}")
    print(f"✅ Found test data: {test_path}")
    
    # Run evaluation
    return quick_test(model_path, test_path)

if __name__ == "__main__":
    # Can be run as script
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick test runner')
    parser.add_argument('--model', type=str, help='Model path')
    parser.add_argument('--test', type=str, help='Test data path')
    parser.add_argument('--output', type=str, default='test_results', help='Output directory')
    
    args = parser.parse_args()
    
    if args.model and args.test:
        # Use provided paths
        summary = quick_test(args.model, args.test, args.output)
    else:
        # Auto-detect files
        summary = test_with_existing_model()
    
    if summary:
        print("\n✅ Evaluation completed successfully!")
    else:
        print("\n❌ Evaluation failed!")
