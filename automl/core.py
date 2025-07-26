import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from sklearn.metrics import accuracy_score
from automl.utils import SimpleTextDataset
from pathlib import Path
from typing import Tuple
from wandb.sdk.wandb_run import Run
from transformers import AutoTokenizer, AutoModel
from automl.model.custom_model import CustomClassificationHead, DistilBertWithCustomHead, freeze_layers
from ray import tune

model_name = 'distilbert-base-uncased'

class TextAutoML:
    def __init__(
        self,
        normalized_class_weights,
        seed,
        token_length,
        epochs,
        batch_size,
        lr,
        weight_decay,
        fraction_layers_to_finetune: float,
        classification_head_hidden_dim: int,
        classification_head_dropout_rate: float,
        classification_head_hidden_layers: int,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        num_classes: int,
        save_path: Path,
        wandb_logger: Run,
        load_path: Path = None,
    ):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.token_length = token_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.normalized_class_weights = normalized_class_weights
        self.fraction_layers_to_finetune = fraction_layers_to_finetune
        self.classification_head_hidden_dim = classification_head_hidden_dim
        self.classification_head_dropout_rate = classification_head_dropout_rate
        self.classification_head_hidden_layers = classification_head_hidden_layers
        self.train_texts = train_df['text'].tolist()
        self.train_labels = train_df['label'].tolist()
        self.val_texts = val_df['text'].tolist()
        self.val_labels = val_df['label'].tolist()
        self.num_classes = num_classes
        self.load_path = load_path
        self.save_path = save_path
        self.wandb_logger = wandb_logger

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size
        wandb_logger.config.update({"vocab_size": self.vocab_size})

        print("---learning rate", self.lr)
        print("---fraction layers to finetune", self.fraction_layers_to_finetune)
        if fraction_layers_to_finetune == 0.0:
            print("---Warning: fraction_layers_to_finetune is set to 0.0, which means all layers will be frozen.")

        # Create model with custom classification head
        self.base_model = AutoModel.from_pretrained(model_name)
        freeze_layers(self.base_model, self.fraction_layers_to_finetune)
        hidden_size = self.base_model.config.hidden_size
        wandb_logger.config.update({"hidden_size": hidden_size})
        self.custom_head = CustomClassificationHead(
            hidden_size=hidden_size,
            num_classes=self.num_classes,
            dropout_rate=self.classification_head_dropout_rate,
            num_hidden_layers=self.classification_head_hidden_layers,
            hidden_dim=self.classification_head_hidden_dim
        )
        # Combine base model and custom head
        self.model = DistilBertWithCustomHead(self.base_model, self.custom_head)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        wandb_logger.config.update({"optimizer": "AdamW"})

    def fit(self):
        """
        Fits a model to the given dataset.
        """
        print("---Loading and preparing data...")

        dataset = SimpleTextDataset(
            self.train_texts, self.train_labels, self.tokenizer, self.token_length
        )
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        _dataset = SimpleTextDataset(
            self.val_texts, self.val_labels, self.tokenizer, self.token_length
        )
        val_loader = DataLoader(_dataset, batch_size=self.batch_size, shuffle=True)

        assert dataset is not None, f"`dataset` cannot be None here!"

        # Training and validating
        val_acc = self._train_loop(
            train_loader,
            val_loader,
            load_path=self.load_path,
            save_path=self.save_path,
        )

        return 1 - val_acc

    def _train_loop(
        self, 
        train_loader: DataLoader,
        val_loader: DataLoader,
        load_path: Path=None,
        save_path: Path=None,
    ):
        # TODO still have not checked if it makes a big difference, should check it with amazon
        if self.normalized_class_weights is not None:
            class_weights = torch.tensor(self.normalized_class_weights, dtype=torch.float).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"---Using class weights: {class_weights}")
        else:
            criterion = nn.CrossEntropyLoss()

        start_epoch = 0
        # handling checkpoint resume
        if load_path is not None:
            _states = torch.load(load_path / "checkpoint.pth", map_location='cpu')  # Load to CPU first
            self.model.load_state_dict(_states["model_state_dict"])
            self.optimizer.load_state_dict(_states["optimizer_state_dict"])
            start_epoch = _states["epoch"]
            print(f"---Resuming from checkpoint at {start_epoch}")

        for epoch in range(start_epoch, self.epochs):            
            total_loss = 0
            train_preds = []
            train_labels_list = []
            
            for batch in train_loader:
                self.model.train()
                self.optimizer.zero_grad()

                inputs = {k: v.to(self.device) for k, v in batch.items()}
                loss, logits = self.model.forward(inputs, criterion)  # Forward pass
                labels = inputs['labels']
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
                # Collect predictions and labels for training accuracy
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=1)
                    train_preds.extend(preds.cpu().numpy())
                    train_labels_list.extend(labels.cpu().numpy())

            # Calculate training accuracy
            train_acc = accuracy_score(train_labels_list, train_preds)
            print(f"---Epoch {epoch + 1}, Loss: {total_loss:.4f}, Train Accuracy: {train_acc:.4f}")

            # Calculate validation accuracy
            val_preds, val_labels = self._predict(val_loader)
            val_acc = accuracy_score(val_labels, val_preds)
            print(f"---Epoch {epoch + 1}, Validation Accuracy: {val_acc:.4f}")

            tune.report(metrics={"val_acc": val_acc})
            
            # Log training and validation accuracy to wandb
            self.wandb_logger.log({
                "epoch": epoch + 1,
                "train_loss": total_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
            })

            # Save checkpoint each epoch
            # TODO can reduce this if it's too much
            if save_path is not None:
                save_path = Path(save_path) if not isinstance(save_path, Path) else save_path
                save_path.mkdir(parents=True, exist_ok=True)

                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch,
                    },
                    save_path / "checkpoint.pth"
                )
        torch.cuda.empty_cache()
        # return the last epoch's val_acc
        return val_acc

    def _predict(self, val_loader: DataLoader):
        self.model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                logits = self.model.predict(inputs)
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()  # Move to CPU here
                preds.extend(batch_preds)
                labels.extend(batch['labels'])
        return np.array(preds), np.array(labels)


    def predict(self, test_data: pd.DataFrame | DataLoader) -> Tuple[np.ndarray, np.ndarray]:

        assert isinstance(test_data, DataLoader) or isinstance(test_data, pd.DataFrame), \
            f"---Input data type: {type(test_data)}; Expected: pd.DataFrame | DataLoader"

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
