import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data(data_path="data/pirvision_office_dataset2.csv"):
    """
    Load and perform comprehensive data exploration
    """
    print("=== PIRvision Dataset Exploration ===\n")
    
    # Load data
    data = pd.read_csv(data_path)
    print(f"Dataset shape: {data.shape}")
    print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
    
    # Basic info
    print("\n=== Basic Dataset Information ===")
    print(data.info())
    
    # Missing values
    print("\n=== Missing Values Analysis ===")
    missing_vals = data.isnull().sum()
    if missing_vals.sum() == 0:
        print("No missing values found in the dataset.")
    else:
        print(f"Total missing values: {missing_vals.sum()}")
        print(missing_vals[missing_vals > 0])
    
    # Target variable analysis
    print("\n=== Target Variable Analysis (Label) ===")
    label_counts = data['Label'].value_counts().sort_index()
    print("Label distribution:")
    for label, count in label_counts.items():
        percentage = (count / len(data)) * 100
        print(f"Label {label}: {count} samples ({percentage:.2f}%)")
    
    # Calculate class imbalance ratio
    majority_class = label_counts.max()
    minority_classes = label_counts[label_counts < majority_class]
    if len(minority_classes) > 0:
        imbalance_ratio = majority_class / minority_classes.min()
        print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1")
        print("This dataset shows significant class imbalance!")
    
    # Temporal analysis
    print("\n=== Temporal Analysis ===")
    data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    data = data.sort_values('DateTime')
    
    print(f"Time range: {data['DateTime'].min()} to {data['DateTime'].max()}")
    print(f"Duration: {data['DateTime'].max() - data['DateTime'].min()}")
    
    # Calculate time intervals
    time_diffs = data['DateTime'].diff().dropna()
    print(f"Average time interval: {time_diffs.mean()}")
    print(f"Median time interval: {time_diffs.median()}")
    
    # Temperature analysis
    print("\n=== Temperature Analysis ===")
    print(f"Temperature range: {data['Temperature_F'].min()}°F to {data['Temperature_F'].max()}°F")
    print(f"Mean temperature: {data['Temperature_F'].mean():.2f}°F")
    print(f"Temperature std: {data['Temperature_F'].std():.2f}°F")
    
    # PIR sensors analysis
    print("\n=== PIR Sensors Analysis ===")
    pir_columns = [col for col in data.columns if col.startswith('PIR_')]
    print(f"Number of PIR sensors: {len(pir_columns)}")
    
    pir_data = data[pir_columns]
    print(f"PIR values range: {pir_data.min().min()} to {pir_data.max().max()}")
    print(f"PIR mean values range: {pir_data.mean().min():.2f} to {pir_data.mean().max():.2f}")
    print(f"PIR std values range: {pir_data.std().min():.2f} to {pir_data.std().max():.2f}")
    
    return data

def create_visualizations(data, save_dir="results"):
    """
    Create comprehensive visualizations
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig_size = (12, 8)
    
    # 1. Label distribution
    plt.figure(figsize=fig_size)
    label_counts = data['Label'].value_counts().sort_index()
    bars = plt.bar(label_counts.index, label_counts.values, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of Occupancy Labels', fontsize=16, fontweight='bold')
    plt.xlabel('Label', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add count labels on bars
    for bar, count in zip(bars, label_counts.values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/len(data)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/label_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Temperature over time by label
    plt.figure(figsize=(15, 6))
    for label in sorted(data['Label'].unique()):
        subset = data[data['Label'] == label]
        plt.scatter(subset['DateTime'], subset['Temperature_F'], 
                   label=f'Label {label}', alpha=0.6, s=20)
    
    plt.title('Temperature Over Time by Occupancy Label', fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Temperature (°F)', fontsize=12)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/temperature_over_time.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. PIR sensor heatmap (correlation)
    pir_columns = [col for col in data.columns if col.startswith('PIR_')]
    pir_sample = data[pir_columns].sample(min(1000, len(data)))  # Sample for performance
    
    plt.figure(figsize=(12, 10))
    correlation_matrix = pir_sample.corr()
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
                square=True, annot=False, cbar_kws={'shrink': 0.8})
    plt.title('PIR Sensors Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/pir_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. PIR sensor values distribution by label
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Select a few representative PIR sensors
    representative_pirs = ['PIR_1', 'PIR_15', 'PIR_30', 'PIR_45']
    
    for i, pir in enumerate(representative_pirs):
        ax = axes[i//2, i%2]
        for label in sorted(data['Label'].unique()):
            subset = data[data['Label'] == label][pir]
            ax.hist(subset, alpha=0.6, bins=30, label=f'Label {label}', density=True)
        
        ax.set_title(f'{pir} Distribution by Label', fontweight='bold')
        ax.set_xlabel(f'{pir} Value')
        ax.set_ylabel('Density')
        ax.legend()
    
    plt.suptitle('PIR Sensor Value Distributions by Occupancy Label', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/pir_distributions_by_label.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Time series plot of selected PIR sensors
    plt.figure(figsize=(15, 8))
    
    # Sample data for visualization (every 10th point)
    sample_data = data.iloc[::10].copy()
    
    for pir in ['PIR_1', 'PIR_15', 'PIR_30']:
        plt.plot(sample_data['DateTime'], sample_data[pir], 
                label=pir, alpha=0.7, linewidth=1)
    
    # Color background by label
    for label in sorted(data['Label'].unique()):
        label_data = sample_data[sample_data['Label'] == label]
        if len(label_data) > 0:
            plt.scatter(label_data['DateTime'], [plt.ylim()[1]] * len(label_data),
                       c=f'C{label}', alpha=0.3, s=10, label=f'Label {label} periods')
    
    plt.title('PIR Sensor Time Series with Occupancy Labels', fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('PIR Value', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/pir_timeseries.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nAll visualizations saved to {save_dir}/ directory")

def analyze_class_imbalance(data):
    """
    Detailed class imbalance analysis
    """
    print("\n=== Detailed Class Imbalance Analysis ===")
    
    label_counts = data['Label'].value_counts().sort_index()
    total_samples = len(data)
    
    print("Class distribution:")
    for label, count in label_counts.items():
        percentage = (count / total_samples) * 100
        print(f"  Label {label}: {count:,} samples ({percentage:.2f}%)")
    
    # Define majority and minority classes
    majority_class = label_counts.idxmax()
    majority_count = label_counts.max()
    
    print(f"\nMajority class: Label {majority_class} with {majority_count:,} samples")
    
    minority_classes = label_counts[label_counts < majority_count]
    if len(minority_classes) > 0:
        print("Minority classes:")
        for label, count in minority_classes.items():
            imbalance_ratio = majority_count / count
            print(f"  Label {label}: {count:,} samples (imbalance ratio: {imbalance_ratio:.2f}:1)")
    
    # Recommendations for handling class imbalance
    print("\n=== Recommendations for Handling Class Imbalance ===")
    max_imbalance = majority_count / minority_classes.min() if len(minority_classes) > 0 else 1
    
    if max_imbalance > 10:
        print("⚠️  SEVERE class imbalance detected (>10:1 ratio)!")
        print("Recommendations:")
        print("  - Use conditional generation to oversample minority classes")
        print("  - Consider SMOTE or other oversampling techniques")
        print("  - Use class-weighted loss functions in models")
        print("  - Focus on precision/recall metrics rather than accuracy")
    elif max_imbalance > 3:
        print("⚠️  Moderate class imbalance detected (>3:1 ratio)")
        print("Recommendations:")
        print("  - Use conditional generation for minority class oversampling")
        print("  - Consider stratified sampling in train/test splits")
    else:
        print("✅ Relatively balanced dataset")
    
    return {
        'label_counts': label_counts,
        'majority_class': majority_class,
        'minority_classes': minority_classes.to_dict(),
        'max_imbalance_ratio': max_imbalance
    }

if __name__ == "__main__":
    # Load and explore data
    data = load_and_explore_data()
    
    # Analyze class imbalance
    imbalance_info = analyze_class_imbalance(data)
    
    # Create visualizations
    create_visualizations(data)
    
    # Save processed data with DateTime
    print("\n=== Saving Processed Data ===")
    data.to_csv("data/pirvision_processed.csv", index=False)
    print("Processed data saved to data/pirvision_processed.csv")