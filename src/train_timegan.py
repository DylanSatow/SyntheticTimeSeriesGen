import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from timegan import TimeGAN, create_data_loader
import os
import matplotlib.pyplot as plt

def load_processed_data(data_dir="data"):
    """Load preprocessed data splits"""
    X_train = np.load(f"{data_dir}/X_train.npy")
    y_train = np.load(f"{data_dir}/y_train.npy")
    X_val = np.load(f"{data_dir}/X_val.npy")
    y_val = np.load(f"{data_dir}/y_val.npy")
    X_test = np.load(f"{data_dir}/X_test.npy")
    y_test = np.load(f"{data_dir}/y_test.npy")
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_unconditional_timegan(X_train, y_train, X_val, y_val, 
                               hidden_dim=128, embedding_dim=64, 
                               autoencoder_epochs=100, adversarial_epochs=200,
                               batch_size=32, device='cpu'):
    """
    Train unconditional TimeGAN model
    """
    print("=== Training Unconditional TimeGAN ===")
    
    # Get data dimensions
    seq_len, input_dim = X_train.shape[1], X_train.shape[2]
    print(f"Sequence length: {seq_len}, Input dimension: {input_dim}")
    
    # Create data loaders
    train_loader = create_data_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = create_data_loader(X_val, y_val, batch_size, shuffle=False)
    
    # Initialize TimeGAN
    timegan = TimeGAN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        device=device
    )
    
    # Phase 1: Train Autoencoder (Embedder + Recovery)
    print("\n--- Phase 1: Autoencoder Training ---")
    train_data_only = [batch[0] for batch in train_loader]  # Only features, no labels
    timegan.train_autoencoder(train_data_only, epochs=autoencoder_epochs)
    
    # Phase 2: Train Adversarial Networks
    print("\n--- Phase 2: Adversarial Training ---")
    timegan.train_adversarial(train_data_only, epochs=adversarial_epochs)
    
    # Save model
    model_path = "models/timegan_unconditional.pth"
    timegan.save_model(model_path)
    
    return timegan

def generate_and_evaluate_samples(timegan, X_test, num_samples=100):
    """
    Generate synthetic samples and perform basic evaluation
    """
    print(f"\n=== Generating {num_samples} Synthetic Samples ===")
    
    seq_len = X_test.shape[1]
    
    # Generate synthetic data
    synthetic_data = timegan.generate_synthetic_data(num_samples, seq_len)
    
    print(f"Generated synthetic data shape: {synthetic_data.shape}")
    print(f"Real data range: [{X_test.min():.3f}, {X_test.max():.3f}]")
    print(f"Synthetic data range: [{synthetic_data.min():.3f}, {synthetic_data.max():.3f}]")
    
    # Save synthetic data
    os.makedirs("results", exist_ok=True)
    np.save("results/synthetic_data_unconditional.npy", synthetic_data)
    
    # Basic visualization
    create_comparison_plots(X_test, synthetic_data)
    
    return synthetic_data

def create_comparison_plots(real_data, synthetic_data, save_dir="results"):
    """
    Create comparison plots between real and synthetic data
    """
    print("Creating comparison visualizations...")
    
    # Select a few samples for visualization
    n_samples = min(5, real_data.shape[0], synthetic_data.shape[0])
    n_features = min(4, real_data.shape[2])  # Show first 4 features
    
    fig, axes = plt.subplots(n_features, 2, figsize=(15, 12))
    
    for feat_idx in range(n_features):
        # Real data plot
        axes[feat_idx, 0].set_title(f'Real Data - Feature {feat_idx+1}')
        for i in range(n_samples):
            axes[feat_idx, 0].plot(real_data[i, :, feat_idx], alpha=0.7)
        axes[feat_idx, 0].set_ylabel('Value')
        axes[feat_idx, 0].grid(True, alpha=0.3)
        
        # Synthetic data plot
        axes[feat_idx, 1].set_title(f'Synthetic Data - Feature {feat_idx+1}')
        for i in range(n_samples):
            axes[feat_idx, 1].plot(synthetic_data[i, :, feat_idx], alpha=0.7)
        axes[feat_idx, 1].set_ylabel('Value')
        axes[feat_idx, 1].grid(True, alpha=0.3)
    
    # Add x-axis labels to bottom plots
    axes[-1, 0].set_xlabel('Time Steps')
    axes[-1, 1].set_xlabel('Time Steps')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/real_vs_synthetic_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Select a few representative features
    feature_indices = [0, 10, 25, 40] if real_data.shape[2] > 40 else [0, 1, 2, 3]
    
    for i, feat_idx in enumerate(feature_indices):
        row, col = i // 2, i % 2
        
        if feat_idx < real_data.shape[2]:
            # Flatten the time series for distribution comparison
            real_values = real_data[:, :, feat_idx].flatten()
            synthetic_values = synthetic_data[:, :, feat_idx].flatten()
            
            axes[row, col].hist(real_values, bins=30, alpha=0.7, label='Real', density=True)
            axes[row, col].hist(synthetic_values, bins=30, alpha=0.7, label='Synthetic', density=True)
            axes[row, col].set_title(f'Feature {feat_idx+1} Distribution')
            axes[row, col].set_xlabel('Value')
            axes[row, col].set_ylabel('Density')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_distributions_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training function"""
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_processed_data()
    
    # Train unconditional TimeGAN
    timegan = train_unconditional_timegan(
        X_train, y_train, X_val, y_val,
        hidden_dim=128,
        embedding_dim=64,
        autoencoder_epochs=50,  # Reduced for faster training
        adversarial_epochs=100,  # Reduced for faster training
        batch_size=16,
        device=device
    )
    
    # Generate and evaluate synthetic samples
    synthetic_data = generate_and_evaluate_samples(timegan, X_test, num_samples=50)
    
    print("\n=== Training Complete ===")
    print("Model saved to: models/timegan_unconditional.pth")
    print("Synthetic data saved to: results/synthetic_data_unconditional.npy")
    print("Visualizations saved to: results/")

if __name__ == "__main__":
    main()