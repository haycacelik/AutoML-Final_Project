import torch.nn as nn
import torch
from pathlib import Path
import json
from transformers import AutoModel

class CustomClassificationHead(nn.Module):
    """Custom classification head with configurable architecture."""
    def __init__(self, hidden_size, num_classes, dropout_rate, num_hidden_layers=1, hidden_dim=None, activation='ReLU', use_layer_norm=False):
        """ Initializes the custom classification head."""
        super().__init__()

        layers = []
        
        # First layer
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_size, hidden_dim))
        layers.append(getattr(nn, activation)())
        
        # Additional hidden layers
        for i in range(num_hidden_layers - 1):
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            # do it for only last layer
            if use_layer_norm and i == num_hidden_layers - 2:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(getattr(nn, activation)())
        
        # Output layer
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)
    
def freeze_layers(model, fraction_layers_to_finetune: float=1.0) -> None:
    """Freeze layers in the model based on fraction to finetune. 
    If the value is 1.0, then do not freeze any layers"""
    
    transformer = None
    print("Freezing transformer layers based on fraction to finetune:", fraction_layers_to_finetune)
    
    # Handle custom head model (DistilBertWithCustomHead)
    if hasattr(model, 'distilbert') and hasattr(model.distilbert, 'transformer'):
        transformer = model.distilbert.transformer
        print("Found transformer in custom head model")
    # Handle direct DistilBERT AutoModel
    elif hasattr(model, 'transformer'):
        transformer = model.transformer
        print("Found transformer in direct AutoModel")
    # Handle AutoModelForSequenceClassification
    elif hasattr(model, 'distilbert') and hasattr(model.distilbert, 'transformer'):
        transformer = model.distilbert.transformer
        print("Found transformer in AutoModelForSequenceClassification")
    else:
        print("Warning: Could not find transformer layers to freeze")
        print(f"Model type: {type(model)}")
        print(f"Model attributes: {dir(model)}")
        if hasattr(model, 'distilbert'):
            print(f"DistilBERT attributes: {dir(model.distilbert)}")
        return
    
    total_layers = len(transformer.layer)
    _num_layers_to_finetune = int(fraction_layers_to_finetune * total_layers)
    layers_to_freeze = total_layers - _num_layers_to_finetune

    print(f"Freezing {layers_to_freeze}/{total_layers} transformer layers")
    
    for layer in transformer.layer[:layers_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False

class DistilBertWithCustomHead(nn.Module):
    """DistilBERT model with custom classification head."""
    def __init__(self, base_model_name, num_classes, dropout_rate, num_hidden_layers, hidden_dim, activation, fraction_layers_to_finetune, use_layer_norm):
        """ Initializes the DistilBERT model with a custom classification head."""
        super().__init__()
        
        # Store configuration
        self.base_model_name = base_model_name
        self.num_classes = num_classes
        self.classification_head_dropout_rate = dropout_rate
        self.classification_head_hidden_layers = num_hidden_layers
        self.classification_head_hidden_dim = hidden_dim
        self.classification_head_activation = activation
        self.fraction_layers_to_finetune = fraction_layers_to_finetune
        self.use_layer_norm = use_layer_norm

        # get the Pre trained DistilBERT model
        self.pre_trained_model = AutoModel.from_pretrained(base_model_name)
        # freeze the layers based on fraction to finetune
        freeze_layers(self.pre_trained_model, fraction_layers_to_finetune)
        # get the hidden size from the pre-trained model
        self.hidden_size = self.pre_trained_model.config.hidden_size
        
        # self.pre_trained_model = base_model
        self.classifier = CustomClassificationHead(
            hidden_size=self.hidden_size,
            num_classes=self.num_classes,
            dropout_rate=self.classification_head_dropout_rate,
            num_hidden_layers=self.classification_head_hidden_layers,
            hidden_dim=self.classification_head_hidden_dim,
            activation=self.classification_head_activation,
            use_layer_norm=use_layer_norm
        )
        
    def forward(self, inputs, loss_function):
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        outputs = self.pre_trained_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Use [CLS] token representation
        cls_output = sequence_output[:, 0]  # First token is [CLS]
        logits = self.classifier(cls_output)
        
        labels = inputs['labels']
        loss = loss_function(logits, labels)
            
        return loss, logits
    
    def predict(self, inputs):
        with torch.no_grad():
            input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
            outputs = self.pre_trained_model(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state
            
            # Use [CLS] token representation
            cls_output = sequence_output[:, 0]
            logits = self.classifier(cls_output)

        return logits
    
    def save_model(self, save_path, filename):
        """Save the complete model including architecture and weights."""
        save_path = Path(save_path)
        best_model_save_path = save_path / "best_model"
        if not best_model_save_path.exists():
            best_model_save_path.mkdir(parents=True, exist_ok=True)

        # If there are already files in the directory, move them to an "old" folder
        if any(best_model_save_path.iterdir()):
            # move the existing files to a old folder
            old_folder = save_path / "old"
            if not old_folder.exists():
                old_folder.mkdir(parents=True, exist_ok=True)
            # Convert to list to avoid modifying directory while iterating
            for file in list(best_model_save_path.iterdir()):
                file.rename(old_folder / file.name)

        # Save the complete model state dict
        model_state = {
            'model_state_dict': self.state_dict(),
            'model_architecture': {
                'base_model_name': self.base_model_name,
                'num_classes': self.num_classes,
                'dropout_rate': self.classification_head_dropout_rate,
                'num_hidden_layers': self.classification_head_hidden_layers,
                'hidden_dim': self.classification_head_hidden_dim,
                'activation': self.classification_head_activation,
                'fraction_layers_to_finetune': self.fraction_layers_to_finetune,
                'use_layer_norm': self.use_layer_norm
            }
        }
        
        # Save model
        model_path = best_model_save_path / f"{filename}.pth"
        torch.save(model_state, model_path)
        
        # Also save just the config as JSON for easy reading
        config_path = best_model_save_path / f"{filename}_config.json"
        with open(config_path, 'w') as f:
            json.dump(model_state['model_architecture'], f, indent=2)
        
        # Clean up model_state to free memory
        del model_state
        
        print(f"Model saved to {model_path}")
        print(f"Config saved to {config_path}")
        return model_path
    
    @classmethod
    def load_model(cls, model_path, device):
        """Reconstruct a saved model. Returns a class instance of DistilBertWithCustomHead."""
        model_path = Path(model_path)
        
        # Load the saved state
        checkpoint = torch.load(model_path, map_location=device)
        model_arch = checkpoint['model_architecture']
        
        # Recreate the model with saved configuration
        model = cls(
            base_model_name=model_arch['base_model_name'],
            num_classes=model_arch['num_classes'],
            dropout_rate=model_arch['dropout_rate'],
            num_hidden_layers=model_arch['num_hidden_layers'],
            hidden_dim=model_arch['hidden_dim'],
            activation=model_arch['activation'],
            fraction_layers_to_finetune=model_arch.get('fraction_layers_to_finetune', 1.0),
            use_layer_norm=model_arch.get('use_layer_norm', False)
        )
        
        # Load the weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Clean up checkpoint to free memory
        del checkpoint
        
        print(f"Model loaded from {model_path}")
        print(f"Architecture: {model_arch['base_model_name']} + Custom Head")
        print(f"Classes: {model_arch['num_classes']}, Hidden: {model_arch['hidden_dim']}, Layers: {model_arch['num_hidden_layers']}")
        
        return model
    