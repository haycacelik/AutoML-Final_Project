import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from automl.utils import SimpleTextDataset
from pathlib import Path
from typing import Tuple
from collections import Counter
import wandb

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class TextAutoML:
    def __init__(
        self,
        normalized_class_weights = None,
        seed=42,
        vocab_size=10000, # Right now does nothing since we use a autotokenizer and use its vocab size
        token_length=128,
        epochs=5,
        batch_size=64,
        lr=1e-4,
        weight_decay=0.0,
        fraction_layers_to_finetune: float=1.0,
    ):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.vocab_size = vocab_size
        self.token_length = token_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.normalized_class_weights = normalized_class_weights

        self.fraction_layers_to_finetune = fraction_layers_to_finetune

        self.model = None
        self.tokenizer = None
        self.vectorizer = None
        self.num_classes = None
        self.train_texts = []
        self.train_labels = []
        self.val_texts = []
        self.val_labels = []

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        num_classes: int,
        vocab_size=None,
        token_length=None,
        epochs=None,
        batch_size=None,
        lr=None,
        weight_decay=None,
        load_path: Path=None,
        save_path: Path=None,
        fraction_layers_to_finetune: float=1.0
    ):
        """
        Fits a model to the given dataset.

        Parameters:
        - train_df (pd.DataFrame): Training data with 'text' and 'label' columns.
        - val_df (pd.DataFrame): Validation data with 'text' and 'label' columns.
        - num_classes (int): Number of classes in the dataset.
        - seed (int): Random seed for reproducibility.
        - vocab_size (int): Maximum vocabulary size.
        - token_length (int): Maximum token sequence length.
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size for training.
        - lr (float): Learning rate.
        - weight_decay (float): Weight decay for optimizer.
        """
        if vocab_size is not None: self.vocab_size = vocab_size
        if token_length is not None: self.token_length = token_length
        if epochs is not None: self.epochs = epochs
        if batch_size is not None: self.batch_size = batch_size
        if lr is not None: self.lr = lr
        if weight_decay is not None: self.weight_decay = weight_decay
        if fraction_layers_to_finetune is not None: self.fraction_layers_to_finetune = fraction_layers_to_finetune
        print("Loading and preparing data...")

        self.train_texts = train_df['text'].tolist()
        self.train_labels = train_df['label'].tolist()
        self.val_texts = val_df['text'].tolist()
        self.val_labels = val_df['label'].tolist()
        self.num_classes = num_classes
        
        train_dist = Counter(self.train_labels)
        val_dist = Counter(self.val_labels)
        print(f"Train class distribution: {train_dist}")
        print(f"Val class distribution: {val_dist}")
        
        # Log class distributions to wandb if wandb is initialized
        if wandb.run is not None:
            wandb.config.update({
                "train_class_distribution": dict(train_dist),
                "val_class_distribution": dict(val_dist)
            })

        dataset = None

        model_name = 'distilbert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size
        dataset = SimpleTextDataset(
            self.train_texts, self.train_labels, self.tokenizer, self.token_length
        )
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        _dataset = SimpleTextDataset(
            self.val_texts, self.val_labels, self.tokenizer, self.token_length
        )
        val_loader = DataLoader(_dataset, batch_size=self.batch_size, shuffle=True)

        if TRANSFORMERS_AVAILABLE:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=self.num_classes
            )
            freeze_layers(self.model, self.fraction_layers_to_finetune)  
        else:
            raise ValueError(
                "Need `AutoTokenizer`, `AutoModelForSequenceClassification` "
                "from `transformers` package."
            )
       
        # Training and validating
        self.model.to(self.device)
        assert dataset is not None, f"`dataset` cannot be None here!"
        # loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        val_acc = self._train_loop(
            train_loader,
            val_loader,
            load_path=load_path,
            save_path=save_path,
        )

        return 1 - val_acc

    def _train_loop(
        self, 
        train_loader: DataLoader,
        val_loader: DataLoader,
        load_path: Path=None,
        save_path: Path=None,
    ):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        if self.normalized_class_weights is not None:
            class_weights = torch.tensor(self.normalized_class_weights, dtype=torch.float).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using class weights: {class_weights}")
        else:
            criterion = nn.CrossEntropyLoss()

        start_epoch = 0
        # handling checkpoint resume
        if load_path is not None:
            _states = torch.load(load_path / "checkpoint.pth", map_location='cpu')  # Load to CPU first
            self.model.load_state_dict(_states["model_state_dict"])
            optimizer.load_state_dict(_states["optimizer_state_dict"])
            start_epoch = _states["epoch"]
            print(f"Resuming from checkpoint at {start_epoch}")

        for epoch in range(start_epoch, self.epochs):            
            total_loss = 0
            train_preds = []
            train_labels_list = []
            
            for batch in train_loader:
                self.model.train()
                optimizer.zero_grad()

                # if isinstance(batch, dict):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs)
                loss = outputs.loss
                labels = inputs['labels']
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                # Collect predictions and labels for training accuracy
                with torch.no_grad():
                    preds = torch.argmax(outputs.logits, dim=1)
                    train_preds.extend(preds.cpu().numpy())
                    train_labels_list.extend(labels.cpu().numpy())

            # Calculate training accuracy
            train_acc = accuracy_score(train_labels_list, train_preds)
            
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            
            # Log training metrics to wandb
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": total_loss,
                    "train_accuracy": train_acc,
                })

            if self.val_texts:
                val_preds, val_labels = self._predict(val_loader)
                val_acc = accuracy_score(val_labels, val_preds)
                print(f"Epoch {epoch + 1}, Validation Accuracy: {val_acc:.4f}")
                
                # Log validation accuracy to wandb
                if wandb.run is not None:
                    wandb.log({
                        "epoch": epoch + 1,
                        "val_accuracy": val_acc,
                    })

        if self.val_texts:
            val_preds, val_labels = self._predict(val_loader)
            val_acc = accuracy_score(val_labels, val_preds)
            print(f"Final Validation Accuracy: {val_acc:.4f}")
            
            # Log final validation accuracy
            if wandb.run is not None:
                wandb.log({
                    "final_val_accuracy": val_acc,
                })

        if save_path is not None:
            save_path = Path(save_path) if not isinstance(save_path, Path) else save_path
            save_path.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                },
                save_path / "checkpoint.pth"
            )   
        torch.cuda.empty_cache()
        return val_acc or 0.0

    def _predict(self, val_loader: DataLoader):
        self.model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                outputs = self.model(**inputs).logits
                labels.extend(batch["labels"])
                preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        
        if isinstance(preds, list):
            preds = [p.item() for p in preds]
            labels = [l.item() for l in labels]
            return np.array(preds), np.array(labels)
        else:
            return preds.cpu().numpy(), labels.cpu().numpy()


    def predict(self, test_data: pd.DataFrame | DataLoader) -> Tuple[np.ndarray, np.ndarray]:

        assert isinstance(test_data, DataLoader) or isinstance(test_data, pd.DataFrame), \
            f"Input data type: {type(test_data)}; Expected: pd.DataFrame | DataLoader"

        if isinstance(test_data, DataLoader):
            return self._predict(test_data)
        
        _dataset = SimpleTextDataset(
            test_data['text'].tolist(),
            test_data['label'].tolist(),
            self.tokenizer,
            self.token_length
        )
        _loader = DataLoader(_dataset, batch_size=self.batch_size, shuffle=True)

        return self._predict(_loader)


def freeze_layers(model, fraction_layers_to_finetune: float=1.0) -> None:
    # if the value is 1.0, then do not freeze any layers
    total_layers = len(model.distilbert.transformer.layer)
    _num_layers_to_finetune = int(fraction_layers_to_finetune * total_layers)
    layers_to_freeze = total_layers - _num_layers_to_finetune

    for layer in model.distilbert.transformer.layer[:layers_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False


# end of file