import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from sklearn.metrics import accuracy_score
from automl.utils import SimpleTextDataset
from pathlib import Path
from typing import Tuple
import wandb
from wandb.sdk.wandb_run import Run
from transformers import AutoTokenizer
from automl.model.custom_model import  DistilBertWithCustomHead
import gc
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
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        save_path: Path,
        wandb_logger: Run = None,
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.token_length = token_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.normalized_class_weights = normalized_class_weights
        self.train_texts = train_df['text'].tolist()
        self.train_labels = train_df['label'].tolist()
        self.val_texts = val_df['text'].tolist()
        self.val_labels = val_df['label'].tolist()
        self.save_path = save_path
        self.wandb_logger = wandb_logger

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.wandb_logger:
            self.wandb_logger.config.update({"vocab_size": self.tokenizer.vocab_size})

        print("---learning rate", self.lr)

    def create_model(self,
                     fraction_layers_to_finetune: float,
                    classification_head_hidden_dim: int,
                    classification_head_dropout_rate: float,
                    classification_head_hidden_layers: int,
                    classification_head_activation: str,
                    num_classes: int,
                    use_layer_norm: bool
                    ):
        """Creates and the model instance."""
        self.model = DistilBertWithCustomHead(
            base_model_name=model_name,
            num_classes=num_classes,
            dropout_rate=classification_head_dropout_rate,
            num_hidden_layers=classification_head_hidden_layers,
            hidden_dim=classification_head_hidden_dim,
            activation=classification_head_activation,
            fraction_layers_to_finetune=fraction_layers_to_finetune,
            use_layer_norm=use_layer_norm
            )

        # Move model to device
        self.model.to(self.device)
        # update the hidden size in wandb config
        hidden_size = self.model.hidden_size
        if self.wandb_logger:
            self.wandb_logger.config.update({"hidden_size": hidden_size})
        # create the optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.wandb_logger:
            self.wandb_logger.config.update({"optimizer": "AdamW"})
        self._model_debug_prints()

    def load_model(self, model_path: Path):
        """Loads a saved model from the specified path."""
        # send the model to the device inside
        self.model = DistilBertWithCustomHead.load_model(model_path, device=self.device)
        # update the hidden size in wandb config
        hidden_size = self.model.hidden_size
        if self.wandb_logger:
            self.wandb_logger.config.update({"hidden_size": hidden_size})
        # create the optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.wandb_logger:
            self.wandb_logger.config.update({"optimizer": "AdamW"})
        self._model_debug_prints()

    def _model_debug_prints(self):
        """Prints debug information about the model."""
        # print the trainable parameters and their shapes
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print("Trainable:", name, param.shape)

        # check if the model takes the correct activation function
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.GELU)):
                print("Activation:", name, module)
            elif isinstance(module, torch.nn.LayerNorm):
                print("LayerNorm:", name, module)

    def fit(self) -> float:
        """
        Fits a model to the given dataset.
        If it is a test run the trial number is set to 0, otherwise it is set to the trial number.
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

        # create the custom head

        # Training and validating
        val_acc = self._train_loop(
            train_loader,
            val_loader,
            load_path=self.load_path,
            save_path=self.save_path,
        )

        # Clear data loaders and datasets to free memory
        del train_loader, val_loader, dataset, _dataset
        gc.collect()  # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return 1 - val_acc

    def _train_loop(
        self, 
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: Path,
    ):
        print("---Starting training loop...")

        # Clear CUDA cache before starting training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # TODO still have not checked if it makes a big difference, should check it with amazon
        if self.normalized_class_weights is not None:
            class_weights = torch.tensor(self.normalized_class_weights, dtype=torch.float).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"---Using class weights: {class_weights}")
        else:
            criterion = nn.CrossEntropyLoss()

        start_epoch = 0
        # handling checkpoint resume
        validation_results = []
        max_validation_accuracy = 0.0

        for epoch in range(start_epoch, self.epochs):            
            total_loss = 0
            train_preds = []
            train_labels_list = []
            
            for batch_idx, batch in enumerate(train_loader):
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

                # Clear variables to free GPU memory
                del inputs, loss, logits, labels, preds

                # Clear CUDA cache every 10 batches to balance memory management and performance
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Calculate training accuracy
            train_acc = accuracy_score(train_labels_list, train_preds)
            print(f"---Epoch {epoch + 1}, Loss: {total_loss:.4f}, Train Accuracy: {train_acc:.4f}")

            # Clear training data from memory
            del train_preds, train_labels_list
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Calculate validation accuracy
            val_preds, val_labels = self._predict(val_loader)
            val_acc = accuracy_score(val_labels, val_preds)
            validation_results.append(val_acc)
            print(f"---Epoch {epoch + 1}, Validation Accuracy: {val_acc:.4f}")

            tune.report(metrics={"val_acc": val_acc})

            # Clear validation data from memory
            del val_preds, val_labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Log training and validation accuracy to wandb
            if self.wandb_logger:
                self.wandb_logger.log({
                    "epoch": epoch + 1,
                    "train_loss": total_loss,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                })

            # check if the validation accuracy is the highest so far
            if val_acc > max_validation_accuracy:
                max_validation_accuracy = val_acc
                print(f"---New best validation accuracy: {max_validation_accuracy:.4f} at epoch {epoch + 1}")
                validation_results = []  # Reset validation results

                # Save the model state
                if save_path is not None:
                    file_name = f"epoch_{epoch + 1}_acc_{val_acc:.4f}"
                    self.model.save_model(save_path, file_name)

            # early stopping, if there has been no improvement for the last 3 epochs
            if len(validation_results) > 3 and all(val <= max_validation_accuracy for val in validation_results[-3:]):
                print(f"---Early stopping at epoch {epoch + 1} due to no improvement in validation accuracy.")
                break

        # Clean up training objects, at the end of training
        del criterion
        if 'class_weights' in locals():
            del class_weights
        gc.collect()  # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # return the last epoch's val_acc
        return val_acc

    def _predict(self, val_loader: DataLoader):
        self.model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                logits = self.model.predict(inputs)
                labels.extend(batch["labels"])
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

                # Clear batch data from GPU memory
                del inputs, logits

                # Clear CUDA cache every 5 batches during validation
                if batch_idx % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if isinstance(preds, list):
            preds = [p.item() for p in preds]
            labels = [l.item() for l in labels]
            return np.array(preds), np.array(labels)
        else:
            return preds.cpu().numpy(), labels.cpu().numpy()


    def predict(self, test_data: pd.DataFrame | DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """to predict with test data, checks test data and makes it a loader fist if it is a DataFrame"""

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

        results = self._predict(_loader)
        # Clean up the dataset and loader to free memory
        del _dataset, _loader
        # since this is for test data, we can clear the memory
        gc.collect()  # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results