import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

class SyntheticDataEvaluator:
    """
    Comprehensive evaluation metrics for synthetic time-series data quality
    """
    
    def __init__(self):
        self.results = {}
    
    def evaluate_fidelity(self, real_data, synthetic_data, feature_names=None):
        """
        Evaluate statistical fidelity between real and synthetic data
        
        Args:
            real_data: Real data [n_samples, seq_len, n_features]
            synthetic_data: Synthetic data [n_samples, seq_len, n_features]
            feature_names: Optional list of feature names
        
        Returns:
            Dictionary with fidelity metrics
        """
        print("=== Evaluating Fidelity (Statistical Similarity) ===")
        
        fidelity_results = {}
        
        # Flatten time series for marginal distribution analysis
        real_flat = real_data.reshape(-1, real_data.shape[-1])
        synthetic_flat = synthetic_data.reshape(-1, synthetic_data.shape[-1])
        
        n_features = real_data.shape[-1]
        
        # 1. Marginal Distribution Similarity
        print("Computing marginal distribution similarity...")
        
        wasserstein_distances = []
        ks_statistics = []
        js_divergences = []
        
        for feat_idx in range(n_features):
            real_feature = real_flat[:, feat_idx]
            synthetic_feature = synthetic_flat[:, feat_idx]
            
            # Wasserstein Distance
            try:
                wd = stats.wasserstein_distance(real_feature, synthetic_feature)
                wasserstein_distances.append(wd)
            except:
                wasserstein_distances.append(np.nan)
            
            # Kolmogorov-Smirnov Test
            try:
                ks_stat, _ = stats.ks_2samp(real_feature, synthetic_feature)
                ks_statistics.append(ks_stat)
            except:
                ks_statistics.append(np.nan)
            
            # Jensen-Shannon Divergence (approximated with histograms)
            try:
                js_div = self._compute_js_divergence(real_feature, synthetic_feature)
                js_divergences.append(js_div)
            except:
                js_divergences.append(np.nan)
        
        fidelity_results['wasserstein_distance'] = {
            'mean': np.nanmean(wasserstein_distances),
            'std': np.nanstd(wasserstein_distances),
            'per_feature': wasserstein_distances
        }
        
        fidelity_results['ks_statistic'] = {
            'mean': np.nanmean(ks_statistics),
            'std': np.nanstd(ks_statistics),
            'per_feature': ks_statistics
        }
        
        fidelity_results['js_divergence'] = {
            'mean': np.nanmean(js_divergences),
            'std': np.nanstd(js_divergences),
            'per_feature': js_divergences
        }
        
        # 2. Correlation Matrix Similarity
        print("Computing correlation matrix similarity...")
        
        real_corr = np.corrcoef(real_flat.T)
        synthetic_corr = np.corrcoef(synthetic_flat.T)
        
        # Frobenius norm of difference
        corr_diff = np.linalg.norm(real_corr - synthetic_corr, 'fro')
        fidelity_results['correlation_difference'] = corr_diff
        
        # 3. Principal Component Analysis Similarity
        print("Computing PCA similarity...")
        
        # Fit PCA on real data
        pca = PCA(n_components=min(10, n_features))
        real_pca = pca.fit_transform(real_flat)
        synthetic_pca = pca.transform(synthetic_flat)
        
        # Compare explained variance ratios
        pca_similarity = np.corrcoef(pca.explained_variance_ratio_, 
                                   np.var(synthetic_pca, axis=0))[0, 1]
        fidelity_results['pca_similarity'] = pca_similarity
        
        # 4. Temporal Autocorrelation Analysis
        print("Computing temporal autocorrelation similarity...")
        
        autocorr_similarities = []
        for feat_idx in range(min(5, n_features)):  # Analyze first 5 features
            real_autocorr = self._compute_autocorrelation(real_data[:, :, feat_idx])
            synthetic_autocorr = self._compute_autocorrelation(synthetic_data[:, :, feat_idx])
            
            # Compute correlation between autocorrelation functions
            if len(real_autocorr) > 0 and len(synthetic_autocorr) > 0:
                min_len = min(len(real_autocorr), len(synthetic_autocorr))
                corr = np.corrcoef(real_autocorr[:min_len], synthetic_autocorr[:min_len])[0, 1]
                if not np.isnan(corr):
                    autocorr_similarities.append(corr)
        
        fidelity_results['autocorrelation_similarity'] = {
            'mean': np.nanmean(autocorr_similarities),
            'std': np.nanstd(autocorr_similarities)
        }
        
        self.results['fidelity'] = fidelity_results
        return fidelity_results
    
    def evaluate_diversity(self, real_data, synthetic_data):
        """
        Evaluate diversity of synthetic data
        
        Args:
            real_data: Real data [n_samples, seq_len, n_features]
            synthetic_data: Synthetic data [n_samples, seq_len, n_features]
        
        Returns:
            Dictionary with diversity metrics
        """
        print("=== Evaluating Diversity ===")
        
        diversity_results = {}
        
        # Flatten for analysis
        real_flat = real_data.reshape(real_data.shape[0], -1)
        synthetic_flat = synthetic_data.reshape(synthetic_data.shape[0], -1)
        
        # 1. Nearest Neighbor Distance
        print("Computing nearest neighbor distances...")
        
        # Find distances from synthetic samples to real samples
        distances = cdist(synthetic_flat, real_flat, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        
        diversity_results['nearest_neighbor_distance'] = {
            'mean': np.mean(min_distances),
            'std': np.std(min_distances),
            'min': np.min(min_distances),
            'max': np.max(min_distances)
        }
        
        # 2. Coverage Analysis
        print("Computing coverage analysis...")
        
        # Use k-nearest neighbors to assess coverage
        k = min(5, len(real_flat))
        nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean')
        nbrs.fit(real_flat)
        
        # For each real sample, check if any synthetic sample is within its k-neighborhood
        coverage_count = 0
        for real_sample in real_flat:
            distances, _ = nbrs.kneighbors([real_sample])
            max_distance = distances[0, -1]  # Distance to k-th neighbor
            
            # Check if any synthetic sample is within this distance
            syn_distances = cdist([real_sample], synthetic_flat, metric='euclidean')[0]
            if np.any(syn_distances <= max_distance):
                coverage_count += 1
        
        coverage_ratio = coverage_count / len(real_flat)
        diversity_results['coverage'] = coverage_ratio
        
        # 3. Intra-class diversity (if multiple synthetic samples)
        print("Computing intra-class diversity...")
        
        if len(synthetic_flat) > 1:
            synthetic_distances = cdist(synthetic_flat, synthetic_flat, metric='euclidean')
            # Remove diagonal (distance to self)
            synthetic_distances = synthetic_distances[np.triu_indices_from(synthetic_distances, k=1)]
            
            diversity_results['intra_synthetic_diversity'] = {
                'mean': np.mean(synthetic_distances),
                'std': np.std(synthetic_distances),
                'min': np.min(synthetic_distances)
            }
        
        self.results['diversity'] = diversity_results
        return diversity_results
    
    def evaluate_utility(self, X_real_train, y_real_train, X_real_test, y_real_test,
                        X_synthetic, y_synthetic=None, X_augmented=None, y_augmented=None):
        """
        Evaluate utility of synthetic data using downstream classification
        
        Args:
            X_real_train, y_real_train: Real training data
            X_real_test, y_real_test: Real test data
            X_synthetic, y_synthetic: Synthetic data (optional labels)
            X_augmented, y_augmented: Augmented dataset (real + synthetic)
        
        Returns:
            Dictionary with utility metrics
        """
        print("=== Evaluating Utility (Downstream Classification) ===")
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
        
        utility_results = {}
        
        # Flatten data for traditional classifiers
        X_real_train_flat = X_real_train.reshape(X_real_train.shape[0], -1)
        X_real_test_flat = X_real_test.reshape(X_real_test.shape[0], -1)
        
        # 1. Baseline: Train on real data only
        print("Training baseline classifier on real data...")
        
        rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_baseline.fit(X_real_train_flat, y_real_train)
        
        y_pred_baseline = rf_baseline.predict(X_real_test_flat)
        
        baseline_metrics = {
            'f1_macro': f1_score(y_real_test, y_pred_baseline, average='macro'),
            'f1_weighted': f1_score(y_real_test, y_pred_baseline, average='weighted'),
            'precision_macro': precision_score(y_real_test, y_pred_baseline, average='macro'),
            'recall_macro': recall_score(y_real_test, y_pred_baseline, average='macro')
        }
        
        utility_results['baseline_real_only'] = baseline_metrics
        
        # 2. Synthetic data only (if labels available)
        if y_synthetic is not None:
            print("Training classifier on synthetic data only...")
            
            X_synthetic_flat = X_synthetic.reshape(X_synthetic.shape[0], -1)
            
            rf_synthetic = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_synthetic.fit(X_synthetic_flat, y_synthetic)
            
            y_pred_synthetic = rf_synthetic.predict(X_real_test_flat)
            
            synthetic_metrics = {
                'f1_macro': f1_score(y_real_test, y_pred_synthetic, average='macro'),
                'f1_weighted': f1_score(y_real_test, y_pred_synthetic, average='weighted'),
                'precision_macro': precision_score(y_real_test, y_pred_synthetic, average='macro'),
                'recall_macro': recall_score(y_real_test, y_pred_synthetic, average='macro')
            }
            
            utility_results['synthetic_only'] = synthetic_metrics
        
        # 3. Augmented dataset (real + synthetic)
        if X_augmented is not None and y_augmented is not None:
            print("Training classifier on augmented dataset...")
            
            X_augmented_flat = X_augmented.reshape(X_augmented.shape[0], -1)
            
            rf_augmented = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_augmented.fit(X_augmented_flat, y_augmented)
            
            y_pred_augmented = rf_augmented.predict(X_real_test_flat)
            
            augmented_metrics = {
                'f1_macro': f1_score(y_real_test, y_pred_augmented, average='macro'),
                'f1_weighted': f1_score(y_real_test, y_pred_augmented, average='weighted'),
                'precision_macro': precision_score(y_real_test, y_pred_augmented, average='macro'),
                'recall_macro': recall_score(y_real_test, y_pred_augmented, average='macro')
            }
            
            utility_results['augmented'] = augmented_metrics
            
            # Calculate improvement
            f1_improvement = augmented_metrics['f1_macro'] - baseline_metrics['f1_macro']
            utility_results['f1_improvement'] = f1_improvement
        
        self.results['utility'] = utility_results
        return utility_results
    
    def create_evaluation_report(self, save_path="results/evaluation_report.txt"):
        """
        Create a comprehensive evaluation report
        """
        print(f"=== Creating Evaluation Report ===")
        
        report = []
        report.append("SYNTHETIC TIME-SERIES DATA EVALUATION REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Fidelity Results
        if 'fidelity' in self.results:
            fidelity = self.results['fidelity']
            report.append("1. FIDELITY (Statistical Similarity)")
            report.append("-" * 35)
            
            if 'wasserstein_distance' in fidelity:
                wd = fidelity['wasserstein_distance']
                report.append(f"Wasserstein Distance: {wd['mean']:.6f} ± {wd['std']:.6f}")
            
            if 'ks_statistic' in fidelity:
                ks = fidelity['ks_statistic']
                report.append(f"KS Statistic: {ks['mean']:.6f} ± {ks['std']:.6f}")
            
            if 'correlation_difference' in fidelity:
                report.append(f"Correlation Matrix Difference: {fidelity['correlation_difference']:.6f}")
            
            if 'autocorrelation_similarity' in fidelity:
                autocorr = fidelity['autocorrelation_similarity']
                report.append(f"Autocorrelation Similarity: {autocorr['mean']:.6f} ± {autocorr['std']:.6f}")
            
            report.append("")
        
        # Diversity Results
        if 'diversity' in self.results:
            diversity = self.results['diversity']
            report.append("2. DIVERSITY")
            report.append("-" * 15)
            
            if 'nearest_neighbor_distance' in diversity:
                nnd = diversity['nearest_neighbor_distance']
                report.append(f"Nearest Neighbor Distance: {nnd['mean']:.6f} ± {nnd['std']:.6f}")
            
            if 'coverage' in diversity:
                report.append(f"Coverage Ratio: {diversity['coverage']:.6f}")
            
            if 'intra_synthetic_diversity' in diversity:
                isd = diversity['intra_synthetic_diversity']
                report.append(f"Intra-Synthetic Diversity: {isd['mean']:.6f} ± {isd['std']:.6f}")
            
            report.append("")
        
        # Utility Results
        if 'utility' in self.results:
            utility = self.results['utility']
            report.append("3. UTILITY (Downstream Classification Performance)")
            report.append("-" * 45)
            
            for dataset_type, metrics in utility.items():
                if dataset_type == 'f1_improvement':
                    continue
                
                report.append(f"{dataset_type.upper()}:")
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        report.append(f"  {metric}: {value:.6f}")
                report.append("")
            
            if 'f1_improvement' in utility:
                improvement = utility['f1_improvement']
                report.append(f"F1-Score Improvement (Augmented vs Baseline): {improvement:.6f}")
                if improvement > 0:
                    report.append("✓ Synthetic data improves classification performance")
                else:
                    report.append("✗ Synthetic data does not improve classification performance")
        
        # Save report
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Evaluation report saved to {save_path}")
        
        # Also print to console
        print("\n".join(report))
    
    def _compute_js_divergence(self, X, Y, bins=50):
        """Compute Jensen-Shannon divergence between two distributions"""
        x_hist, _ = np.histogram(X, bins=bins, density=True)
        y_hist, _ = np.histogram(Y, bins=bins, density=True)
        
        # Normalize
        x_hist = x_hist / np.sum(x_hist)
        y_hist = y_hist / np.sum(y_hist)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        x_hist = x_hist + epsilon
        y_hist = y_hist + epsilon
        
        # Jensen-Shannon divergence
        m = 0.5 * (x_hist + y_hist)
        js_div = 0.5 * stats.entropy(x_hist, m) + 0.5 * stats.entropy(y_hist, m)
        
        return js_div
    
    def _compute_autocorrelation(self, data, max_lags=20):
        """Compute autocorrelation function for time series data"""
        autocorrs = []
        
        for i in range(data.shape[0]):  # For each sample
            series = data[i, :]
            autocorr = []
            
            for lag in range(1, min(max_lags, len(series))):
                if len(series) > lag:
                    corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                    if not np.isnan(corr):
                        autocorr.append(corr)
            
            if len(autocorr) > 0:
                autocorrs.append(np.mean(autocorr))
        
        return autocorrs