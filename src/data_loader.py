import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import os
import urllib.request
import zipfile
from pathlib import Path

def download_pirvision_dataset(data_dir="data"):
    """
    Download the PIRvision dataset from UCI ML Repository
    """
    try:
        # Try using the official UCI ML repository API
        pirvision = fetch_ucirepo(id=1101)
        
        # Extract features and targets
        X = pirvision.data.features
        y = pirvision.data.targets
        
        # Combine features and targets
        data = pd.concat([X, y], axis=1)
        
        # Save to CSV
        os.makedirs(data_dir, exist_ok=True)
        data.to_csv(f"{data_dir}/pirvision_raw.csv", index=False)
        
        print(f"Dataset downloaded successfully to {data_dir}/pirvision_raw.csv")
        print(f"Dataset shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        return data
        
    except Exception as e:
        print(f"Error downloading with ucimlrepo: {e}")
        
        # Alternative download method using direct URL
        print("Attempting alternative download method...")
        try:
            # Direct download from UCI repository
            url = "https://archive.ics.uci.edu/static/public/1101/pirvision_fog_presence_detection.zip"
            zip_path = f"{data_dir}/pirvision_dataset.zip"
            
            os.makedirs(data_dir, exist_ok=True)
            
            print("Downloading dataset...")
            urllib.request.urlretrieve(url, zip_path)
            
            # Extract zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            # Clean up zip file
            os.remove(zip_path)
            
            print(f"Dataset downloaded and extracted to {data_dir}/")
            
            # Find and load the main data file
            data_files = list(Path(data_dir).glob("*.csv"))
            if data_files:
                data = pd.read_csv(data_files[0])
                print(f"Loaded data from {data_files[0]}")
                print(f"Dataset shape: {data.shape}")
                print(f"Columns: {list(data.columns)}")
                return data
            else:
                print("No CSV files found in extracted data")
                return None
                
        except Exception as e2:
            print(f"Alternative download also failed: {e2}")
            return None

def load_pirvision_data(data_path="data/pirvision_raw.csv"):
    """
    Load the PIRvision dataset from local file
    """
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
        print(f"Loaded dataset from {data_path}")
        print(f"Dataset shape: {data.shape}")
        return data
    else:
        print(f"File not found: {data_path}")
        print("Please run download_pirvision_dataset() first")
        return None

if __name__ == "__main__":
    # Download the dataset
    data = download_pirvision_dataset()
    
    if data is not None:
        print("\nFirst few rows:")
        print(data.head())
        print("\nDataset info:")
        print(data.info())
        print("\nDataset description:")
        print(data.describe())