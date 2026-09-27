"""
Script to download data from Kaggle.
"""
import os
import kaggle

def download_kaggle_dataset(dataset_name, path):
    """
    Download dataset from Kaggle.

    Args:
        dataset_name (str): Name of the dataset on Kaggle
        path (str): Local path to save the dataset
    """
    os.makedirs(path, exist_ok=True)
    kaggle.api.dataset_download_files(dataset_name, path=path, unzip=True)
    print(f"Downloaded {dataset_name} to {path}")

if __name__ == "__main__":
    # Example usage
    download_kaggle_dataset("username/bearing-dataset", "data/raw/")