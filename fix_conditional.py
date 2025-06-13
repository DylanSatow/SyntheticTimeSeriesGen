"""
Quick fix for conditional TimeGAN label mapping issue
"""
import numpy as np

# Check the label distribution
y_train = np.load("data/y_train.npy")
print("Original labels:", np.unique(y_train))

# Map labels to start from 0
unique_labels = np.unique(y_train)
label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
reverse_mapping = {new_label: old_label for old_label, new_label in label_mapping.items()}

print("Label mapping:", label_mapping)

# Apply mapping to all label files
for split in ['train', 'val', 'test']:
    y_file = f"data/y_{split}.npy"
    y_data = np.load(y_file)
    y_mapped = np.array([label_mapping[label] for label in y_data])
    np.save(f"data/y_{split}_mapped.npy", y_mapped)
    print(f"Mapped {split} labels: {np.unique(y_data)} -> {np.unique(y_mapped)}")

# Save mapping for later use
np.save("data/label_mapping.npy", label_mapping)
np.save("data/reverse_mapping.npy", reverse_mapping)

print("✅ Label mapping complete!")