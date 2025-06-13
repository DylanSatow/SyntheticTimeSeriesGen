import numpy as np
import torch
from conditional_timegan import ConditionalTimeGAN, create_conditional_data_loader
import matplotlib.pyplot as plt
import os

def load_processed_data(data_dir="data"):
    """Load preprocessed data splits"""
    X_train = np.load(f"{data_dir}/X_train.npy")
    y_train = np.load(f"{data_dir}/y_train.npy")
    X_val = np.load(f"{data_dir}/X_val.npy")
    y_val = np.load(f"{data_dir}/y_val.npy")
    X_test = np.load(f"{data_dir}/X_test.npy")
    y_test = np.load(f"{data_dir}/y_test.npy")
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    
    # Print class distribution
    unique_classes, counts = np.unique(y_train, return_counts=True)
    print("Training class distribution:")
    for cls, count in zip(unique_classes, counts):
        print(f"  Class {cls}: {count} samples ({count/len(y_train)*100:.1f}%)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def calculate_target_counts(y_train, balance_strategy='oversample_to_majority'):
    """
    Calculate target counts for balanced dataset generation
    
    Args:
        y_train: Training labels
        balance_strategy: Strategy for balancing ('oversample_to_majority', 'equal_distribution')
    
    Returns:
        Dictionary with target counts for each class
    """
    unique_classes, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique_classes, counts))
    
    if balance_strategy == 'oversample_to_majority':
        # Oversample minority classes to match majority class count
        target_count = max(counts)
        target_counts = {}
        for cls in unique_classes:
            current_count = class_counts[cls]
            if current_count < target_count:
                # Generate additional samples for minority class
                target_counts[cls] = target_count - current_count
            else:
                # No additional samples needed for majority class
                target_counts[cls] = 0
    
    elif balance_strategy == 'equal_distribution':
        # Generate equal number of samples for all classes
        target_count = max(counts)
        target_counts = {cls: target_count for cls in unique_classes}
    
    print(f"\nTarget synthetic samples per class ({balance_strategy}):")
    for cls, count in target_counts.items():
        print(f"  Class {cls}: {count} synthetic samples")
    
    return target_counts

def train_conditional_timegan(X_train, y_train, X_val, y_val,
                            hidden_dim=64, embedding_dim=32,
                            autoencoder_epochs=50, adversarial_epochs=100,
                            batch_size=16, device='cpu'):
    """
    Train conditional TimeGAN model for class-conditional generation
    """
    print("=== Training Conditional TimeGAN ===")
    
    # Get data dimensions
    seq_len, input_dim = X_train.shape[1], X_train.shape[2]
    num_classes = len(np.unique(y_train))
    
    print(f"Sequence length: {seq_len}")
    print(f"Input dimension: {input_dim}")
    print(f"Number of classes: {num_classes}")
    
    # Create data loaders
    train_loader = create_conditional_data_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = create_conditional_data_loader(X_val, y_val, batch_size, shuffle=False)
    
    # Initialize Conditional TimeGAN
    conditional_timegan = ConditionalTimeGAN(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        device=device
    )
    
    # Phase 1: Train Autoencoder
    print("\n--- Phase 1: Conditional Autoencoder Training ---")
    conditional_timegan.train_autoencoder(train_loader, epochs=autoencoder_epochs)
    
    # Phase 2: Train Adversarial Networks
    print("\n--- Phase 2: Conditional Adversarial Training ---")
    conditional_timegan.train_adversarial(train_loader, epochs=adversarial_epochs)
    
    # Save model
    model_path = "models/conditional_timegan.pth"
    conditional_timegan.save_model(model_path)
    
    return conditional_timegan

def generate_balanced_synthetic_data(conditional_timegan, y_train, seq_len, 
                                   balance_strategy='oversample_to_majority'):
    """
    Generate synthetic data to balance the dataset
    """
    print(f"\n=== Generating Balanced Synthetic Data ===")
    
    # Calculate target counts
    target_counts = calculate_target_counts(y_train, balance_strategy)
    
    # Generate synthetic data for each class
    synthetic_data = conditional_timegan.generate_balanced_dataset(target_counts, seq_len)
    
    # Save synthetic data
    os.makedirs("results", exist_ok=True)
    for class_label, data in synthetic_data.items():
        if data.shape[0] > 0:  # Only save if there are samples to generate
            np.save(f"results/synthetic_class_{class_label}.npy", data)
            print(f"Saved {data.shape[0]} synthetic samples for class {class_label}")
    
    return synthetic_data

def create_augmented_dataset(X_train, y_train, synthetic_data):
    """
    Create augmented dataset by combining real and synthetic data
    """
    print("\n=== Creating Augmented Dataset ===")
    
    augmented_X = [X_train]
    augmented_y = [y_train]
    
    total_synthetic_samples = 0
    
    for class_label, syn_data in synthetic_data.items():
        if syn_data.shape[0] > 0:
            # Create labels for synthetic data
            syn_labels = np.full(syn_data.shape[0], class_label)
            
            augmented_X.append(syn_data)
            augmented_y.append(syn_labels)
            
            total_synthetic_samples += syn_data.shape[0]
    
    # Concatenate all data
    X_augmented = np.concatenate(augmented_X, axis=0)
    y_augmented = np.concatenate(augmented_y, axis=0)
    
    print(f"Original dataset size: {len(X_train)}")
    print(f"Synthetic samples added: {total_synthetic_samples}")
    print(f"Augmented dataset size: {len(X_augmented)}")
    
    # Print new class distribution
    unique_classes, counts = np.unique(y_augmented, return_counts=True)
    print("\nAugmented dataset class distribution:")
    for cls, count in zip(unique_classes, counts):
        print(f"  Class {cls}: {count} samples ({count/len(y_augmented)*100:.1f}%)")
    
    # Save augmented dataset
    np.save("data/X_augmented.npy", X_augmented)
    np.save("data/y_augmented.npy", y_augmented)
    print("\nAugmented dataset saved to data/X_augmented.npy and data/y_augmented.npy")
    
    return X_augmented, y_augmented

def visualize_class_specific_generation(real_data, real_labels, synthetic_data, 
                                      target_class=1, save_dir="results"):
    """
    Visualize synthetic data generation for a specific class
    """
    print(f"\nCreating visualizations for class {target_class}...")
    
    # Get real data for target class
    class_mask = real_labels == target_class
    real_class_data = real_data[class_mask]
    
    # Get synthetic data for target class
    if target_class in synthetic_data and synthetic_data[target_class].shape[0] > 0:
        syn_class_data = synthetic_data[target_class]
        
        # Create comparison plot
        n_samples = min(3, real_class_data.shape[0], syn_class_data.shape[0])
        n_features = min(4, real_class_data.shape[2])
        
        fig, axes = plt.subplots(n_features, 2, figsize=(15, 12))
        
        for feat_idx in range(n_features):
            # Real data plot
            axes[feat_idx, 0].set_title(f'Real Data - Class {target_class} - Feature {feat_idx+1}')
            for i in range(n_samples):
                axes[feat_idx, 0].plot(real_class_data[i, :, feat_idx], alpha=0.7)
            axes[feat_idx, 0].set_ylabel('Value')
            axes[feat_idx, 0].grid(True, alpha=0.3)
            
            # Synthetic data plot
            axes[feat_idx, 1].set_title(f'Synthetic Data - Class {target_class} - Feature {feat_idx+1}')
            for i in range(n_samples):
                axes[feat_idx, 1].plot(syn_class_data[i, :, feat_idx], alpha=0.7)
            axes[feat_idx, 1].set_ylabel('Value')
            axes[feat_idx, 1].grid(True, alpha=0.3)
        
        axes[-1, 0].set_xlabel('Time Steps')
        axes[-1, 1].set_xlabel('Time Steps')
        
        plt.suptitle(f'Real vs Synthetic Data Comparison - Class {target_class}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/class_{target_class}_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print(f"No synthetic data generated for class {target_class}")

def main():
    """Main training function for conditional TimeGAN"""
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_processed_data()
    
    # Train conditional TimeGAN with reduced parameters for faster training
    conditional_timegan = train_conditional_timegan(
        X_train, y_train, X_val, y_val,
        hidden_dim=64,
        embedding_dim=32,
        autoencoder_epochs=30,  # Reduced for faster training
        adversarial_epochs=50,   # Reduced for faster training
        batch_size=16,
        device=device
    )
    
    # Generate balanced synthetic data
    seq_len = X_train.shape[1]
    synthetic_data = generate_balanced_synthetic_data(
        conditional_timegan, y_train, seq_len, 
        balance_strategy='oversample_to_majority'
    )
    
    # Create augmented dataset
    X_augmented, y_augmented = create_augmented_dataset(X_train, y_train, synthetic_data)
    
    # Visualize results for minority classes
    for target_class in [1, 3]:  # Minority classes from our analysis
        visualize_class_specific_generation(
            X_train, y_train, synthetic_data, target_class
        )
    
    print("\n=== Conditional TimeGAN Training Complete ===")
    print("Model saved to: models/conditional_timegan.pth")
    print("Synthetic data saved to: results/synthetic_class_*.npy")
    print("Augmented dataset saved to: data/X_augmented.npy and data/y_augmented.npy")

if __name__ == "__main__":
    main()