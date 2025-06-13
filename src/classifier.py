import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class TimeSeriesClassifier(nn.Module):
    """
    Simple LSTM-based classifier for time-series occupancy detection
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2):
        super(TimeSeriesClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input sequences [batch_size, seq_len, input_dim]
        
        Returns:
            Class logits [batch_size, num_classes]
        """
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output for classification
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_dim * 2]
        
        # Classification
        logits = self.classifier(last_output)
        
        return logits

class ClassifierTrainer:
    """
    Trainer class for the time-series classifier
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        
    def train(self, train_loader, val_loader=None, epochs=50, class_weights=None):
        """
        Train the classifier
        """
        if class_weights is not None:
            # Apply class weights to loss function
            weights = torch.FloatTensor(list(class_weights.values())).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0
            total_samples = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
            
            avg_loss = epoch_loss / total_samples
            train_losses.append(avg_loss)
            
            # Validation phase
            if val_loader is not None:
                val_acc = self.evaluate(val_loader)
                val_accuracies.append(val_acc)
                
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}')
            else:
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}')
            
            self.scheduler.step()
        
        return train_losses, val_accuracies
    
    def evaluate(self, test_loader):
        """
        Evaluate the classifier
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        return accuracy
    
    def predict(self, test_loader):
        """
        Make predictions
        """
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(target.cpu().numpy())
        
        return np.array(predictions), np.array(true_labels)
    
    def get_detailed_metrics(self, test_loader, class_names=None):
        """
        Get detailed classification metrics
        """
        predictions, true_labels = self.predict(test_loader)
        
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(len(np.unique(true_labels)))]
        
        # Classification report
        report = classification_report(true_labels, predictions, 
                                     target_names=class_names, 
                                     output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        return report, cm

def create_data_loader(X, y, batch_size=32, shuffle=False):
    """
    Create data loader for training/testing
    """
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader

def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def compare_classifier_performance():
    """
    Compare classifier performance on different datasets
    """
    print("=== Comparing Classifier Performance ===")
    
    # Load data
    X_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train.npy")
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")
    
    # Check for augmented data
    try:
        X_augmented = np.load("data/X_augmented.npy")
        y_augmented = np.load("data/y_augmented.npy")
        has_augmented = True
    except:
        has_augmented = False
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = X_train.shape[2]
    num_classes = len(np.unique(y_train))
    
    results = {}
    
    # 1. Train on original data
    print("\n1. Training on original data...")
    model1 = TimeSeriesClassifier(input_dim, num_classes=num_classes)
    trainer1 = ClassifierTrainer(model1, device)
    
    train_loader1 = create_data_loader(X_train, y_train, batch_size=16, shuffle=True)
    test_loader = create_data_loader(X_test, y_test, batch_size=16, shuffle=False)
    
    trainer1.train(train_loader1, epochs=30)
    
    report1, cm1 = trainer1.get_detailed_metrics(test_loader, 
                                                class_names=[f'Class_{i}' for i in range(num_classes)])
    results['original'] = report1
    
    print(f"Original Data - Test Accuracy: {report1['accuracy']:.4f}")
    print(f"Original Data - Macro F1: {report1['macro avg']['f1-score']:.4f}")
    
    # 2. Train on augmented data (if available)
    if has_augmented:
        print("\n2. Training on augmented data...")
        model2 = TimeSeriesClassifier(input_dim, num_classes=num_classes)
        trainer2 = ClassifierTrainer(model2, device)
        
        train_loader2 = create_data_loader(X_augmented, y_augmented, batch_size=16, shuffle=True)
        
        trainer2.train(train_loader2, epochs=30)
        
        report2, cm2 = trainer2.get_detailed_metrics(test_loader,
                                                    class_names=[f'Class_{i}' for i in range(num_classes)])
        results['augmented'] = report2
        
        print(f"Augmented Data - Test Accuracy: {report2['accuracy']:.4f}")
        print(f"Augmented Data - Macro F1: {report2['macro avg']['f1-score']:.4f}")
        
        # Calculate improvement
        acc_improvement = report2['accuracy'] - report1['accuracy']
        f1_improvement = report2['macro avg']['f1-score'] - report1['macro avg']['f1-score']
        
        print(f"\nImprovement with synthetic data:")
        print(f"  Accuracy: {acc_improvement:+.4f} ({acc_improvement*100:+.2f}%)")
        print(f"  Macro F1: {f1_improvement:+.4f} ({f1_improvement*100:+.2f}%)")
        
        # Plot confusion matrices
        plot_confusion_matrix(cm1, [f'Class_{i}' for i in range(num_classes)], 
                            'results/confusion_matrix_original.png')
        plot_confusion_matrix(cm2, [f'Class_{i}' for i in range(num_classes)], 
                            'results/confusion_matrix_augmented.png')
    
    return results

if __name__ == "__main__":
    results = compare_classifier_performance()