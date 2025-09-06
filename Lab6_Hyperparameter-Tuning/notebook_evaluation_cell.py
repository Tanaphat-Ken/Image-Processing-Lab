# Competition Evaluation Cell for Lab6 Notebook
# =============================================
# Add this cell to your Lab6_hypertune.ipynb to run competition evaluation

# Import the quick test function
import sys
import os
sys.path.append('.')

from quick_test import quick_test

# Run competition evaluation
print("🏆 Running Competition Evaluation...")
print("=" * 50)

# Test with your best model
results = quick_test(
    model_path="best_autoencoder.pth",  # Your best trained model
    test_zip="test.zip",                # Competition test data
    output_dir="test_results"           # Results directory
)

if results:
    print("\n🎯 Key Competition Metrics:")
    print(f"📊 Total Test Images: {results['total_images']}")
    print(f"🔥 Average PSNR: {results['avg_psnr']:.2f} dB")
    print(f"✨ Average SSIM: {results['avg_ssim']:.3f}")
    print(f"📉 Average MSE: {results['avg_mse']:.6f}")
    
    print("\n📁 Check these files for detailed results:")
    print("   - test_results/competition_submission.json")
    print("   - test_results/detailed_results.csv") 
    print("   - test_results/metrics_distribution.png")
    print("   - test_results/sample_results.png")
    
    print("\n🏆 Competition Summary:")
    print(f"   Your model achieved {results['avg_psnr']:.2f} dB PSNR")
    print(f"   and {results['avg_ssim']:.3f} SSIM on {results['total_images']} test images!")
else:
    print("❌ Evaluation failed. Check the error messages above.")
