#!/usr/bin/env python3
"""
Main execution script for Synthetic Time-Series Data Generation project
Provides a complete pipeline from data preprocessing to evaluation
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

def setup_project():
    """Create necessary directories and setup project structure"""
    directories = ['data', 'models', 'results', 'notebooks']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
    print("✅ Project directories created")

def run_data_pipeline():
    """Execute complete data pipeline"""
    print("\n" + "="*60)
    print("🔄 RUNNING DATA PIPELINE")
    print("="*60)
    
    # Step 1: Download and explore data
    print("\n📥 Step 1: Downloading and exploring data...")
    try:
        from data_loader import download_pirvision_dataset
        from data_exploration import load_and_explore_data, analyze_class_imbalance, create_visualizations
        
        # Download dataset
        data = download_pirvision_dataset()
        
        if data is not None:
            # Explore data
            data = load_and_explore_data()
            analyze_class_imbalance(data)
            create_visualizations(data)
            print("✅ Data exploration completed")
        else:
            print("❌ Failed to download dataset")
            return False
            
    except Exception as e:
        print(f"❌ Data exploration failed: {e}")
        return False
    
    # Step 2: Preprocess data
    print("\n🔧 Step 2: Preprocessing data...")
    try:
        from data_preprocessing import PIRDataPreprocessor
        
        preprocessor = PIRDataPreprocessor(
            sequence_length=60,
            overlap=0.5,
            normalization='minmax'
        )
        
        splits, class_weights = preprocessor.full_preprocessing_pipeline()
        print("✅ Data preprocessing completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        return False

def run_training_pipeline(quick_mode=False):
    """Execute model training pipeline"""
    print("\n" + "="*60)
    print("🚀 RUNNING TRAINING PIPELINE")
    print("="*60)
    
    # Fix label mapping first
    print("\n🔧 Fixing label mapping...")
    try:
        import numpy as np
        
        # Load and map labels
        y_train = np.load("data/y_train.npy")
        unique_labels = np.unique(y_train)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        
        # Apply mapping to all splits
        for split in ['train', 'val', 'test']:
            y_data = np.load(f"data/y_{split}.npy")
            y_mapped = np.array([label_mapping[int(label)] for label in y_data])
            np.save(f"data/y_{split}_mapped.npy", y_mapped)
        
        print("✅ Label mapping completed")
        
    except Exception as e:
        print(f"❌ Label mapping failed: {e}")
        return False
    
    # Train conditional TimeGAN
    print("\n🤖 Training Conditional TimeGAN...")
    try:
        from train_conditional_timegan import main as train_conditional_main
        
        # Override training parameters for quick mode
        if quick_mode:
            print("⚡ Quick mode: Using reduced training parameters")
            # This would require modifying the training script to accept parameters
            # For now, we'll use the test pipeline
            from test_pipeline import quick_test
            quick_test()
            print("✅ Quick training completed")
        else:
            # Run full training (this takes significant time)
            print("⏳ Full training mode (this may take a while...)")
            train_conditional_main()
            print("✅ Full training completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def run_evaluation_pipeline():
    """Execute evaluation pipeline"""
    print("\n" + "="*60)
    print("📊 RUNNING EVALUATION PIPELINE")
    print("="*60)
    
    try:
        from run_evaluation import run_comprehensive_evaluation
        
        evaluator = run_comprehensive_evaluation()
        print("✅ Evaluation completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        print("Make sure synthetic data has been generated first")
        return False

def run_downstream_evaluation():
    """Run downstream classification evaluation"""
    print("\n🎯 Running downstream classification evaluation...")
    try:
        from classifier import compare_classifier_performance
        
        results = compare_classifier_performance()
        print("✅ Downstream evaluation completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Downstream evaluation failed: {e}")
        return False

def create_project_summary():
    """Create a summary of project results"""
    print("\n" + "="*60)
    print("📋 PROJECT SUMMARY")
    print("="*60)
    
    summary = []
    summary.append("🔬 SYNTHETIC TIME-SERIES DATA GENERATION PROJECT")
    summary.append("="*50)
    summary.append("")
    
    # Check what has been completed
    completed_steps = []
    
    if os.path.exists("data/pirvision_processed.csv"):
        completed_steps.append("✅ Data exploration and preprocessing")
    
    if os.path.exists("models/conditional_timegan.pth"):
        completed_steps.append("✅ Conditional TimeGAN training")
    
    if os.path.exists("results/evaluation_report.txt"):
        completed_steps.append("✅ Comprehensive evaluation")
    
    if os.path.exists("data/X_augmented.npy"):
        completed_steps.append("✅ Balanced dataset generation")
    
    summary.append("COMPLETED STEPS:")
    for step in completed_steps:
        summary.append(f"  {step}")
    
    summary.append("")
    summary.append("📊 KEY METRICS:")
    
    # Try to read evaluation results
    try:
        if os.path.exists("results/evaluation_report.txt"):
            with open("results/evaluation_report.txt", 'r') as f:
                content = f.read()
                summary.append("  (See results/evaluation_report.txt for detailed metrics)")
        else:
            summary.append("  Run evaluation pipeline to generate metrics")
    except:
        summary.append("  Evaluation metrics not yet available")
    
    summary.append("")
    summary.append("📁 OUTPUT FILES:")
    
    output_files = [
        ("data/X_augmented.npy", "Augmented dataset (real + synthetic)"),
        ("results/synthetic_class_*.npy", "Class-specific synthetic data"),
        ("results/evaluation_report.txt", "Comprehensive evaluation report"),
        ("models/conditional_timegan.pth", "Trained TimeGAN model"),
        ("results/*.png", "Visualization plots")
    ]
    
    for file_pattern, description in output_files:
        summary.append(f"  {file_pattern} - {description}")
    
    summary.append("")
    summary.append("🚀 NEXT STEPS:")
    summary.append("  • Scale up training with GPU for better results")
    summary.append("  • Experiment with different hyperparameters")
    summary.append("  • Apply to other imbalanced time-series datasets")
    summary.append("  • Implement privacy evaluation (membership inference)")
    
    # Print and save summary
    summary_text = "\n".join(summary)
    print(summary_text)
    
    with open("PROJECT_SUMMARY.md", 'w') as f:
        f.write(summary_text)
    
    print("\n📄 Summary saved to PROJECT_SUMMARY.md")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Synthetic Time-Series Data Generation Pipeline"
    )
    parser.add_argument(
        '--pipeline', 
        choices=['all', 'data', 'train', 'eval', 'summary'],
        default='all',
        help='Which pipeline to run'
    )
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Use quick mode for faster testing'
    )
    
    args = parser.parse_args()
    
    print("🎯 SYNTHETIC TIME-SERIES DATA GENERATION")
    print("=" * 60)
    print("Addressing class imbalance in occupancy detection")
    print("with conditional TimeGAN synthetic data generation")
    print("=" * 60)
    
    # Setup project
    setup_project()
    
    success = True
    
    if args.pipeline in ['all', 'data']:
        success &= run_data_pipeline()
    
    if args.pipeline in ['all', 'train'] and success:
        success &= run_training_pipeline(quick_mode=args.quick)
    
    if args.pipeline in ['all', 'eval'] and success:
        success &= run_evaluation_pipeline()
        success &= run_downstream_evaluation()
    
    if args.pipeline in ['all', 'summary']:
        create_project_summary()
    
    print("\n" + "="*60)
    if success:
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("✅ Check the results/ directory for outputs")
        print("📊 Review PROJECT_SUMMARY.md for overview")
    else:
        print("❌ PIPELINE ENCOUNTERED ERRORS")
        print("🔍 Check error messages above for troubleshooting")
    print("="*60)

if __name__ == "__main__":
    main()