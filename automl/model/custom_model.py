import torch.nn as nn

class CustomClassificationHead(nn.Module):
    """Custom classification head with configurable architecture."""
    def __init__(self, hidden_size, num_classes, dropout_rate, num_hidden_layers=1, hidden_dim=None):
        super().__init__()
        
        layers = []
        
        # First layer
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_size, hidden_dim))
        layers.append(nn.ReLU())
        
        # Additional hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)
    
def freeze_layers(model, fraction_layers_to_finetune: float=1.0) -> None:
    """Freeze layers in the model based on fraction to finetune."""
    # if the value is 1.0, then do not freeze any layers
    
    # Handle custom head model
    if hasattr(model, 'distilbert') and hasattr(model.distilbert, 'transformer'):
        transformer = model.distilbert.transformer
    # Handle standard transformers model
    elif hasattr(model, 'distilbert'):
        transformer = model.distilbert.transformer
    else:
        print("Warning: Could not find transformer layers to freeze")
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
    def __init__(self, base_model, custom_head):
        super().__init__()
        self.distilbert = base_model
        self.classifier = custom_head
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Use [CLS] token representation
        cls_output = sequence_output[:, 0]  # First token is [CLS]
        logits = self.classifier(cls_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return type('Outputs', (), {'loss': loss, 'logits': logits})()

