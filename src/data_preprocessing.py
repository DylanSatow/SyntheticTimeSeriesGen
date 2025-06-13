import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import os

class PIRDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for PIRvision dataset
    """
    
    def __init__(self, sequence_length=60, overlap=0.5, normalization='minmax'):
        """
        Initialize the preprocessor
        
        Args:
            sequence_length: Length of each sequence for time-series modeling
            overlap: Overlap ratio for sliding window (0.0 to 1.0)
            normalization: 'minmax', 'standard', or 'none'
        """
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.normalization = normalization
        self.scalers = {}
        self.feature_columns = None
        self.label_mapping = None
        
    def load_data(self, data_path="data/pirvision_processed.csv"):
        """Load the processed PIRvision data"""
        self.data = pd.read_csv(data_path)
        print(f"Loaded data with shape: {self.data.shape}")
        
        # Convert DateTime
        if 'DateTime' not in self.data.columns:
            self.data['DateTime'] = pd.to_datetime(self.data['Date'] + ' ' + self.data['Time'])
        
        # Sort by time
        self.data = self.data.sort_values('DateTime').reset_index(drop=True)
        
        return self.data
    
    def identify_features(self):
        """Identify feature columns and target column"""
        # PIR sensor columns
        pir_columns = [col for col in self.data.columns if col.startswith('PIR_')]
        
        # Temperature column
        temp_columns = ['Temperature_F']
        
        # Combine all feature columns
        self.feature_columns = pir_columns + temp_columns
        
        print(f"Identified {len(self.feature_columns)} feature columns:")
        print(f"  - {len(pir_columns)} PIR sensors")
        print(f"  - {len(temp_columns)} temperature features")
        
        return self.feature_columns
    
    def handle_outliers(self, method='iqr', factor=1.5):
        """
        Handle outliers in the data
        
        Args:
            method: 'iqr' or 'zscore'
            factor: IQR factor or z-score threshold
        """
        print(f"\n=== Handling Outliers ({method} method) ===")
        
        original_shape = self.data.shape
        
        if method == 'iqr':
            for col in self.feature_columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                # Cap outliers instead of removing them to preserve temporal sequence
                self.data[col] = np.clip(self.data[col], lower_bound, upper_bound)
        
        elif method == 'zscore':
            from scipy import stats
            for col in self.feature_columns:
                z_scores = np.abs(stats.zscore(self.data[col]))
                # Cap outliers
                outlier_mask = z_scores > factor
                if outlier_mask.sum() > 0:
                    median_val = self.data[col].median()
                    self.data.loc[outlier_mask, col] = median_val
        
        print(f"Outlier handling completed. Shape unchanged: {self.data.shape}")
        
    def normalize_features(self):
        """Normalize feature values"""
        print(f"\n=== Normalizing Features ({self.normalization}) ===")
        
        if self.normalization == 'none':
            print("No normalization applied")
            return
        
        # Initialize scalers for different feature groups
        if self.normalization == 'minmax':
            self.scalers['pir'] = MinMaxScaler()
            self.scalers['temp'] = MinMaxScaler()
        elif self.normalization == 'standard':
            self.scalers['pir'] = StandardScaler()
            self.scalers['temp'] = StandardScaler()
        
        # PIR sensors
        pir_columns = [col for col in self.feature_columns if col.startswith('PIR_')]
        if pir_columns:
            self.data[pir_columns] = self.scalers['pir'].fit_transform(self.data[pir_columns])
            print(f"Normalized {len(pir_columns)} PIR sensor columns")
        
        # Temperature
        temp_columns = [col for col in self.feature_columns if col in ['Temperature_F']]
        if temp_columns:
            self.data[temp_columns] = self.scalers['temp'].fit_transform(self.data[temp_columns])
            print(f"Normalized {len(temp_columns)} temperature columns")
        
    def create_sequences(self, data_subset=None):
        """
        Create sequences using sliding window approach
        
        Args:
            data_subset: Specific subset of data to create sequences from
        """
        print(f"\n=== Creating Sequences ===")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Overlap ratio: {self.overlap}")
        
        if data_subset is None:
            data_subset = self.data.copy()
        
        features = data_subset[self.feature_columns].values
        labels = data_subset['Label'].values
        timestamps = data_subset['DateTime'].values
        
        # Calculate step size for sliding window
        step_size = max(1, int(self.sequence_length * (1 - self.overlap)))
        
        sequences = []
        sequence_labels = []
        sequence_timestamps = []
        
        for i in range(0, len(features) - self.sequence_length + 1, step_size):
            # Extract sequence
            seq = features[i:i + self.sequence_length]
            
            # Get label for the sequence (use the last label in the sequence)
            seq_label = labels[i + self.sequence_length - 1]
            
            # Get timestamp for the sequence
            seq_timestamp = timestamps[i + self.sequence_length - 1]
            
            sequences.append(seq)
            sequence_labels.append(seq_label)
            sequence_timestamps.append(seq_timestamp)
        
        sequences = np.array(sequences)
        sequence_labels = np.array(sequence_labels)
        sequence_timestamps = np.array(sequence_timestamps)
        
        print(f"Created {len(sequences)} sequences")
        print(f"Sequence shape: {sequences.shape}")
        print(f"Labels shape: {sequence_labels.shape}")
        
        return sequences, sequence_labels, sequence_timestamps
    
    def train_test_split(self, test_size=0.2, val_size=0.2, stratify=True, random_state=42):
        """
        Split data into train, validation, and test sets
        
        Args:
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            stratify: Whether to stratify by label
            random_state: Random seed
        """
        print(f"\n=== Splitting Data ===")
        
        # Create sequences from the full dataset
        X, y, timestamps = self.create_sequences()
        
        # First split: separate test set
        if stratify:
            X_temp, X_test, y_temp, y_test, ts_temp, ts_test = train_test_split(
                X, y, timestamps, test_size=test_size, stratify=y, random_state=random_state
            )
        else:
            X_temp, X_test, y_temp, y_test, ts_temp, ts_test = train_test_split(
                X, y, timestamps, test_size=test_size, random_state=random_state
            )
        
        # Second split: separate validation from training
        if stratify:
            X_train, X_val, y_train, y_val, ts_train, ts_val = train_test_split(
                X_temp, y_temp, ts_temp, test_size=val_size, stratify=y_temp, random_state=random_state
            )
        else:
            X_train, X_val, y_train, y_val, ts_train, ts_val = train_test_split(
                X_temp, y_temp, ts_temp, test_size=val_size, random_state=random_state
            )
        
        print(f"Training set: {X_train.shape[0]} sequences")
        print(f"Validation set: {X_val.shape[0]} sequences")
        print(f"Test set: {X_test.shape[0]} sequences")
        
        # Print class distribution
        print("\nClass distribution:")
        for split_name, labels in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            unique, counts = np.unique(labels, return_counts=True)
            dist = {f'Label_{u}': f'{c} ({c/len(labels)*100:.1f}%)' for u, c in zip(unique, counts)}
            print(f"  {split_name}: {dist}")
        
        return {
            'X_train': X_train, 'y_train': y_train, 'ts_train': ts_train,
            'X_val': X_val, 'y_val': y_val, 'ts_val': ts_val,
            'X_test': X_test, 'y_test': y_test, 'ts_test': ts_test
        }
    
    def save_preprocessor(self, save_path="models/preprocessor.pkl"):
        """Save the preprocessor configuration and scalers"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        preprocessor_config = {
            'sequence_length': self.sequence_length,
            'overlap': self.overlap,
            'normalization': self.normalization,
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'label_mapping': self.label_mapping
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(preprocessor_config, f)
        
        print(f"Preprocessor saved to {save_path}")
    
    def load_preprocessor(self, load_path="models/preprocessor.pkl"):
        """Load the preprocessor configuration and scalers"""
        with open(load_path, 'rb') as f:
            config = pickle.load(f)
        
        self.sequence_length = config['sequence_length']
        self.overlap = config['overlap']
        self.normalization = config['normalization']
        self.scalers = config['scalers']
        self.feature_columns = config['feature_columns']
        self.label_mapping = config['label_mapping']
        
        print(f"Preprocessor loaded from {load_path}")
    
    def get_class_weights(self, y):
        """Calculate class weights for handling imbalanced data"""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y
        )
        
        weight_dict = dict(zip(classes, class_weights))
        print(f"Class weights: {weight_dict}")
        
        return weight_dict
    
    def full_preprocessing_pipeline(self, data_path="data/pirvision_processed.csv", 
                                   save_data=True, save_dir="data"):
        """
        Run the complete preprocessing pipeline
        """
        print("=== Starting Full Preprocessing Pipeline ===\n")
        
        # Load data
        self.load_data(data_path)
        
        # Identify features
        self.identify_features()
        
        # Handle outliers
        self.handle_outliers()
        
        # Normalize features
        self.normalize_features()
        
        # Split data
        splits = self.train_test_split()
        
        # Calculate class weights
        class_weights = self.get_class_weights(splits['y_train'])
        
        # Save preprocessor
        self.save_preprocessor()
        
        if save_data:
            # Save processed data splits
            os.makedirs(save_dir, exist_ok=True)
            
            np.save(f"{save_dir}/X_train.npy", splits['X_train'])
            np.save(f"{save_dir}/y_train.npy", splits['y_train'])
            np.save(f"{save_dir}/X_val.npy", splits['X_val'])
            np.save(f"{save_dir}/y_val.npy", splits['y_val'])
            np.save(f"{save_dir}/X_test.npy", splits['X_test'])
            np.save(f"{save_dir}/y_test.npy", splits['y_test'])
            
            print(f"\nProcessed data splits saved to {save_dir}/")
        
        print("\n=== Preprocessing Pipeline Completed ===")
        
        return splits, class_weights

def main():
    """Main preprocessing function"""
    
    # Initialize preprocessor with optimized parameters for time-series
    preprocessor = PIRDataPreprocessor(
        sequence_length=60,  # 60 time steps (about 10-20 minutes of data)
        overlap=0.5,         # 50% overlap for better data utilization
        normalization='minmax'  # MinMax scaling for neural networks
    )
    
    # Run full preprocessing pipeline
    splits, class_weights = preprocessor.full_preprocessing_pipeline()
    
    # Print final summary
    print("\n=== Final Summary ===")
    print(f"Feature dimensions: {splits['X_train'].shape[1:]} (time_steps, features)")
    print(f"Number of features: {splits['X_train'].shape[2]}")
    print(f"Sequence length: {splits['X_train'].shape[1]}")
    print(f"Total sequences: {splits['X_train'].shape[0] + splits['X_val'].shape[0] + splits['X_test'].shape[0]}")
    
    return preprocessor, splits, class_weights

if __name__ == "__main__":
    preprocessor, splits, class_weights = main()