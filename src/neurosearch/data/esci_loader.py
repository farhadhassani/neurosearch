import os
import pandas as pd
from sklearn.model_selection import train_test_split
import kagglehub

class ESCILoader:
    """Utility class to download, load, and preprocess the Amazon ESCI dataset.
    The actual download URLs are placeholders and should be replaced with the real source.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)



    def download_dataset(self) -> str:
        """Download the ESCI dataset from GitHub and return the local path."""
        repo_url = "https://github.com/amazon-science/esci-data.git"
        target_dir = os.path.join(self.data_dir, "esci-data")
        
        if os.path.exists(target_dir):
            print(f"Dataset directory {target_dir} already exists. Skipping download.")
            return target_dir
            
        print(f"Cloning Amazon ESCI dataset from {repo_url}...")
        import subprocess
        try:
            subprocess.run(["git", "clone", "--depth", "1", repo_url, target_dir], check=True)
            print(f"Dataset cloned to: {target_dir}")
            return target_dir
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone dataset: {e}")
            raise

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load the ESCI CSV file into a DataFrame."""
        return pd.read_csv(csv_path)

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        """Split the DataFrame into train/val/test sets."""
        train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)
        train, val = train_test_split(train_val, test_size=val_size / (1 - test_size), random_state=random_state)
        return train, val, test

    def map_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map ESCI categorical labels to numeric scores.
        ESCI labels: E, S, C, I -> scores: 3, 2, 1, 0
        """
        label_map = {"E": 3, "S": 2, "C": 1, "I": 0}
        df = df.copy()
        df["esci_score"] = df["esci_label"].map(label_map)
        return df
