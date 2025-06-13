import numpy as np
import os
from evaluation_metrics import SyntheticDataEvaluator
import glob

def load_all_data():
    """Load all datasets for evaluation"""
    print("=== Loading Data for Evaluation ===")
    
    # Load original datasets
    X_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train.npy")
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")
    
    print(f"Real training data: {X_train.shape}")
    print(f"Real test data: {X_test.shape}")
    
    # Load synthetic data if available
    synthetic_data = {}
    synthetic_files = glob.glob("results/synthetic_class_*.npy")
    
    if synthetic_files:
        print("Found synthetic data files:")
        for file in synthetic_files:
            class_label = int(file.split('_')[-1].replace('.npy', ''))
            data = np.load(file)
            if data.shape[0] > 0:  # Only load if not empty
                synthetic_data[class_label] = data
                print(f"  Class {class_label}: {data.shape}")
    
    # Combine synthetic data if available
    X_synthetic = None
    y_synthetic = None
    
    if synthetic_data:
        X_synthetic_list = []
        y_synthetic_list = []
        
        for class_label, data in synthetic_data.items():
            X_synthetic_list.append(data)
            y_synthetic_list.append(np.full(data.shape[0], class_label))
        
        if X_synthetic_list:
            X_synthetic = np.concatenate(X_synthetic_list, axis=0)
            y_synthetic = np.concatenate(y_synthetic_list, axis=0)
            print(f"Combined synthetic data: {X_synthetic.shape}")
    
    # Load augmented data if available
    X_augmented = None
    y_augmented = None
    
    if os.path.exists("data/X_augmented.npy") and os.path.exists("data/y_augmented.npy"):
        X_augmented = np.load("data/X_augmented.npy")
        y_augmented = np.load("data/y_augmented.npy")
        print(f"Augmented data: {X_augmented.shape}")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'X_synthetic': X_synthetic, 'y_synthetic': y_synthetic,
        'X_augmented': X_augmented, 'y_augmented': y_augmented
    }

def run_comprehensive_evaluation():
    """Run complete evaluation pipeline"""
    print("=== Starting Comprehensive Evaluation ===\n")
    
    # Load data
    data = load_all_data()
    
    if data['X_synthetic'] is None:
        print("No synthetic data found. Please run the conditional TimeGAN training first.")
        return
    
    # Initialize evaluator
    evaluator = SyntheticDataEvaluator()
    
    # 1. Evaluate Fidelity
    print("\n" + "="*60)
    fidelity_results = evaluator.evaluate_fidelity(
        real_data=data['X_train'],
        synthetic_data=data['X_synthetic']
    )
    
    # 2. Evaluate Diversity
    print("\n" + "="*60)
    diversity_results = evaluator.evaluate_diversity(
        real_data=data['X_train'],
        synthetic_data=data['X_synthetic']
    )
    
    # 3. Evaluate Utility
    print("\n" + "="*60)
    utility_results = evaluator.evaluate_utility(
        X_real_train=data['X_train'],
        y_real_train=data['y_train'],
        X_real_test=data['X_test'],
        y_real_test=data['y_test'],
        X_synthetic=data['X_synthetic'],
        y_synthetic=data['y_synthetic'],
        X_augmented=data['X_augmented'],
        y_augmented=data['y_augmented']
    )
    
    # 4. Create comprehensive report
    print("\n" + "="*60)
    evaluator.create_evaluation_report("results/evaluation_report.txt")
    
    # 5. Create summary insights
    create_evaluation_summary(fidelity_results, diversity_results, utility_results)
    
    return evaluator

def create_evaluation_summary(fidelity_results, diversity_results, utility_results):
    """Create a high-level summary of evaluation results"""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    # Fidelity Summary
    print("\n🔍 FIDELITY ASSESSMENT:")
    if 'wasserstein_distance' in fidelity_results:
        wd_mean = fidelity_results['wasserstein_distance']['mean']
        if wd_mean < 0.1:
            print("  ✅ Excellent statistical similarity (Low Wasserstein Distance)")
        elif wd_mean < 0.3:
            print("  ⚠️  Good statistical similarity (Moderate Wasserstein Distance)")
        else:
            print("  ❌ Poor statistical similarity (High Wasserstein Distance)")
    
    if 'correlation_difference' in fidelity_results:
        corr_diff = fidelity_results['correlation_difference']
        if corr_diff < 1.0:
            print("  ✅ Excellent correlation structure preservation")
        elif corr_diff < 3.0:
            print("  ⚠️  Good correlation structure preservation")
        else:
            print("  ❌ Poor correlation structure preservation")
    
    # Diversity Summary
    print("\n🎯 DIVERSITY ASSESSMENT:")
    if 'coverage' in diversity_results:
        coverage = diversity_results['coverage']
        if coverage > 0.8:
            print("  ✅ Excellent coverage of real data space")
        elif coverage > 0.6:
            print("  ⚠️  Good coverage of real data space")
        else:
            print("  ❌ Poor coverage of real data space")
    
    if 'nearest_neighbor_distance' in diversity_results:
        nnd_mean = diversity_results['nearest_neighbor_distance']['mean']
        print(f"  📊 Average distance to real data: {nnd_mean:.4f}")
    
    # Utility Summary
    print("\n🎯 UTILITY ASSESSMENT:")
    if 'f1_improvement' in utility_results:
        improvement = utility_results['f1_improvement']
        if improvement > 0.05:
            print(f"  ✅ Significant performance improvement: +{improvement:.1%}")
        elif improvement > 0.01:
            print(f"  ⚠️  Moderate performance improvement: +{improvement:.1%}")
        elif improvement > -0.01:
            print(f"  ⚠️  Minimal impact: {improvement:+.1%}")
        else:
            print(f"  ❌ Performance degradation: {improvement:+.1%}")
    
    # Overall Assessment
    print("\n🏆 OVERALL ASSESSMENT:")
    
    # Calculate overall score
    scores = []
    
    # Fidelity score (lower is better for Wasserstein distance)
    if 'wasserstein_distance' in fidelity_results:
        wd = fidelity_results['wasserstein_distance']['mean']
        fidelity_score = max(0, 1 - wd / 0.5)  # Normalize to 0-1 range
        scores.append(fidelity_score)
    
    # Diversity score
    if 'coverage' in diversity_results:
        diversity_score = diversity_results['coverage']
        scores.append(diversity_score)
    
    # Utility score
    if 'f1_improvement' in utility_results:
        improvement = utility_results['f1_improvement']
        utility_score = max(0, min(1, improvement * 10 + 0.5))  # Normalize improvement
        scores.append(utility_score)
    
    if scores:
        overall_score = np.mean(scores)
        if overall_score > 0.8:
            print("  ✅ EXCELLENT synthetic data quality")
        elif overall_score > 0.6:
            print("  ⚠️  GOOD synthetic data quality")
        elif overall_score > 0.4:
            print("  ⚠️  MODERATE synthetic data quality")
        else:
            print("  ❌ POOR synthetic data quality")
        
        print(f"  📊 Overall Quality Score: {overall_score:.2f}/1.00")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if 'wasserstein_distance' in fidelity_results and fidelity_results['wasserstein_distance']['mean'] > 0.3:
        print("  • Consider tuning TimeGAN hyperparameters to improve statistical fidelity")
    
    if 'coverage' in diversity_results and diversity_results['coverage'] < 0.6:
        print("  • Increase diversity in generation or adjust noise distribution")
    
    if 'f1_improvement' in utility_results and utility_results['f1_improvement'] < 0:
        print("  • Review class balancing strategy and generation quality")
        print("  • Consider different downstream model architectures")
    
    print("  • Monitor synthetic data for potential overfitting to training set")
    print("  • Validate results on additional downstream tasks")

def main():
    """Main evaluation function"""
    try:
        evaluator = run_comprehensive_evaluation()
        print("\n" + "="*60)
        print("✅ Evaluation completed successfully!")
        print("📄 Detailed report saved to: results/evaluation_report.txt")
        print("="*60)
        
        return evaluator
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {str(e)}")
        print("Please ensure that:")
        print("  1. Synthetic data has been generated")
        print("  2. All required data files are present")
        print("  3. Required dependencies are installed")
        
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()