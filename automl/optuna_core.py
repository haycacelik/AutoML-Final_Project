import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from sklearn.metrics import accuracy_score
from automl.utils import SimpleTextDataset
from pathlib import Path
from typing import Tuple
from transformers import AutoTokenizer
from automl.model.custom_model import  DistilBertWithCustomHead
import gc
import os 
import logging

# Set up logging for Ray Tune
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('training.log')  # File output
    ]
)
logger = logging.getLogger(__name__) 

model_name = 'distilbert-base-uncased'

class TextAutoML:
    def __init__(
        self,
        normalized_class_weights,
        seed,
        token_length,
        max_epochs,
        batch_size,
        lr,
        weight_decay,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.token_length = token_length
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.normalized_class_weights = normalized_class_weights
        self.train_texts = train_df['text'].tolist()
        self.train_labels = train_df['label'].tolist()
        self.val_texts = val_df['text'].tolist()
        self.val_labels = val_df['label'].tolist()
        self.train_accuracies = []
        self.val_accuracies = []

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

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

        # create the optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # check if the model is correctly created
        # self._model_debug_prints()

        self.overfit = False
        self.max_validation_accuracy = 0.0
        self.starting_epoch = 0 
        self.no_improvement_count = 0  # Counter for early stopping


    def load_model(self, temp_dir: Path):
        """Loads a saved model from the specified path."""
        print("---Loading model from", temp_dir)
        
        state = torch.load(temp_dir / "extra_state.pth")
        self.starting_epoch = state["epoch"] + 1
        self.overfit = state["overfit"]
        self.max_validation_accuracy = state["max_validation_accuracy"]

        # if the model is not overfitting, load the model and optimizer
        if not self.overfit:
            self.no_improvement_count = state["no_improvement_count"]

            # send the model to the device inside
            self.model = DistilBertWithCustomHead.load_model(temp_dir / "model.pth", device=self.device)

            # load the optimizer state
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.optimizer.load_state_dict(torch.load(temp_dir / "optimizer.pth"))

            # check if the model is correctly loaded
            # self._model_debug_prints()

        if self.starting_epoch >= self.max_epochs:
            print(f"---Model has already been trained for {self.max_epochs} epochs, starting from epoch {self.starting_epoch}")
            raise ValueError(f"---Model has already been trained for {self.max_epochs} epochs")

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
    

    def fit(self, save_dir) -> float:
        """
        Fits a model to the given dataset.
        If it is a test run the trial number is set to 0, otherwise it is set to the trial number.
        """
        if self.overfit:
            # TODO check if self.max_epochs-1 is true
            self.save_extra_info(current_epoch=self.max_epochs - 1, save_dir=Path(save_dir))
            return [], 1 - self.max_validation_accuracy

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
        val_accuracies = self._train_loop(
            train_loader,
            val_loader,
            save_dir=save_dir
        )

        # Clear data loaders and datasets to free memory
        del train_loader, val_loader, dataset, _dataset
        gc.collect()  # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return val_accuracies, 1 - self.max_validation_accuracy

    def _train_loop(
        self, 
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: str
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

        val_accuracies = []
        for epoch in range(self.starting_epoch, self.max_epochs):

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

            # Clear training data from memory
            del train_preds, train_labels_list
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Calculate validation accuracy
            val_preds, val_labels = self._predict(val_loader)
            val_acc = accuracy_score(val_labels, val_preds)

            # # for debugging purposes
            # if epoch < 3 :
            #     val_acc = self.max_validation_accuracy + 0.01
            # else:
            #     val_acc = 0.0
            # train_acc = 0.5
            
            logger.info(f"Epoch {epoch + 1}, Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")
            print(f"---Epoch {epoch + 1}, Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")

            # Clear validation data from memory
            del val_preds, val_labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # check if the validation accuracy is the highest so far
            if val_acc > self.max_validation_accuracy:
                self.max_validation_accuracy = val_acc
                self.no_improvement_count = 0  # Reset no improvement count
                val_accuracies.append((epoch, val_acc))  # Append the best validation accuracy
                print(f"---Saving new best model at epoch {epoch + 1} with validation accuracy {val_acc:.4f}")
                self.model.save_model(save_dir=save_dir/"best_version")
                self.save_optimizer(save_dir=save_dir/"best_version")
                self.save_extra_info(current_epoch=epoch, save_dir=Path(save_dir) / "best_version")
            else:
                self.no_improvement_count += 1
                print(f"---No improvement in validation accuracy for {self.no_improvement_count} epochs.")
                if self.no_improvement_count >= 3:
                    print(f"---Early stopping at epoch {epoch + 1} due to no improvement in validation accuracy.")
                    self.overfit = True

                    # save the extra info
                    self.save_extra_info(current_epoch=self.max_epochs - 1, save_dir=Path(save_dir))

                    # Clean up training objects, at the end of training
                    del criterion
                    if 'class_weights' in locals():
                        del class_weights
                    # Force garbage collection to free memory
                    gc.collect() 
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return val_accuracies

        # TODO save the model
        self.model.save_model(save_dir=save_dir)
        self.save_optimizer(save_dir=save_dir)
        self.save_extra_info(current_epoch=epoch, save_dir=Path(save_dir))

        # doesnt usually do anything here anyways
        # Clean up training objects, at the end of training
        del criterion
        if 'class_weights' in locals():
            del class_weights
        # Force garbage collection to free memory
        gc.collect() 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # return the last epoch's val_acc

        print("val_accuracies", val_accuracies)
        return val_accuracies

    
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

    def save_optimizer(self, save_dir):
        """Saves the optimizer state."""
        save_dir = Path(save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        optimizer_save_path = save_dir / "optimizer.pth"
        torch.save(self.optimizer.state_dict(), optimizer_save_path)
        print(f"---Optimizer state saved to {optimizer_save_path}")

    def save_extra_info(self, current_epoch: int, save_dir: Path):
        # if path does not exist, create it
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": current_epoch,
            "max_validation_accuracy": self.max_validation_accuracy,
            "overfit": self.overfit,
            "no_improvement_count": self.no_improvement_count if not self.overfit else 0,
            }, os.path.join(save_dir, "extra_state.pth"))
