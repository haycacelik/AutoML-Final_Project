"""Dataset classes for NLP AutoML tasks."""
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import string
from sklearn.model_selection import train_test_split


class BaseTextDataset(ABC):
    """Base class for text datasets."""
    
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = Path(data_path) if isinstance(data_path, str) else data_path
        self.vocab_size = 10000  # Default vocab size
        self.max_length = 512    # Default max sequence length
        
    @abstractmethod
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test data."""
        pass
    
    def get_num_classes(self, train_df) -> int:
        """Return number of classes."""
        return train_df['label'].nunique()

    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def create_dataloaders(
        self,
        val_size: float = 0.2,
        random_state: int = 42,
        set_class_count_min: bool = False,
        use_class_weights: bool = True
    ) -> Dict[str, Any]:
        """Create train/validation/test dataloaders and preprocessing objects."""
        train_df, test_df = self.load_data()  # not implemented in base class `BaseTextDataset`
        label_column = train_df.columns[1]
        print("Train_df with labels and class instance counts:\n",train_df[label_column].value_counts())
        
        if set_class_count_min and use_class_weights:
            raise ValueError("Cannot set class count to minimum and use class weights at the same time. Or neither of them.")
        
        if set_class_count_min:
            # get class counts, sort by label from 0 to 
            label_column = train_df.columns[1]
            class_counts = self.get_num_classes(train_df)
            print("train_df before split, original class counts", class_counts)
            
            class_instance_counts = train_df[label_column].value_counts().sort_index().to_numpy()
            print("train_df before split, original class counts", class_counts)
            
            # find the minimum number of instances in any class
            min_class_count = class_instance_counts.min()

            # Subsample each class to the minimum class count
            subsampled_dfs = []
            for label, group in train_df.groupby(label_column):
                subsampled = group.sample(n=min_class_count, random_state=random_state)
                subsampled_dfs.append(subsampled)
            train_df = pd.concat(subsampled_dfs)
            print(f"New dataset length: {len(train_df)}")
            print(train_df[label_column].value_counts())

        # Split training data into train/validation
        if val_size > 0:
            train_df, val_df = train_test_split(
                train_df, test_size=val_size, random_state=random_state,
                stratify=train_df['label'] if 'label' in train_df.columns else None
            )
        else:
            val_df = None

        if use_class_weights:
            # how many classes are there? find the 
            label_column = train_df.columns[1]
            class_counts = self.get_num_classes(train_df)
            print("train_df after split, original class counts", class_counts)
            
            class_instance_counts = train_df[label_column].value_counts().sort_index().to_numpy()
            print("train_df after split, original class counts", class_instance_counts)
            
            # Calculate class weights
            # for each class do total instances / instances for that class
            total_instances = len(train_df)
            class_weights = total_instances / class_instance_counts
            print("class weights", class_weights, type(class_weights))
            normalized_class_weights = class_weights * (len(class_instance_counts) / np.sum(class_weights))
            print("normalized class weights", normalized_class_weights, type(normalized_class_weights))
        
        # Preprocess text
        train_df['text'] = train_df['text'].apply(self.preprocess_text)
        if val_df is not None:
            val_df['text'] = val_df['text'].apply(self.preprocess_text)
        test_df['text'] = test_df['text'].apply(self.preprocess_text)

        return {
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            "num_classes": self.get_num_classes(train_df),
            "normalized_class_weights": normalized_class_weights if use_class_weights else None
        }


class AGNewsDataset(BaseTextDataset):
    """AG News dataset for news categorization (4 classes)."""
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load AG News data."""
        # This assumes CSV files with columns: label, text
        train_path = self.data_path / "ag_news" / "train.csv"
        test_path = self.data_path / "ag_news" / "test.csv"
        
        if train_path.exists() and test_path.exists():
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        else:
            raise FileNotFoundError(f"Data files not found at {train_path}, generating dummy data...")
        
        return train_df, test_df


class IMDBDataset(BaseTextDataset):
    """IMDB movie review sentiment dataset (2 classes)."""
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load IMDB data."""
        train_path = self.data_path / "imdb" / "train.csv"
        test_path = self.data_path / "imdb" / "test.csv"
        
        if train_path.exists() and test_path.exists():
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        else:
            raise FileNotFoundError(f"Data files not found at {train_path}, generating dummy data...")
        
        return train_df, test_df


class AmazonReviewsDataset(BaseTextDataset):
    """Amazon product reviews dataset (3 classes)."""
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load Amazon reviews data."""
        train_path = self.data_path / "amazon" / "train.csv"
        test_path = self.data_path / "amazon" / "test.csv"
        
        if train_path.exists() and test_path.exists():
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        else:
            raise FileNotFoundError(f"Data files not found at {train_path}, generating dummy data...")
        
        return train_df, test_df


class DBpediaDataset(BaseTextDataset):
    """DBpedia ontology classification dataset (14 classes)."""
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load DBpedia ontology data."""
        train_path = self.data_path / "dbpedia" / "train.csv"
        test_path = self.data_path / "dbpedia" / "test.csv"
        
        if train_path.exists() and test_path.exists():
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        else:
            raise FileNotFoundError(f"Data files not found at {train_path}, generating dummy data...")

        # Crucial handling of negative class label
        class_num = self.get_num_classes(train_df)
        train_df['label'] = train_df['label'].replace(-1, class_num - 1)
        test_df['label'] = test_df['label'].replace(-1, class_num - 1)

        return train_df, test_df

def set_class_num_to_lowest():
    pass