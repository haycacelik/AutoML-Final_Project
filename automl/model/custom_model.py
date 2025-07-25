import torch.nn as nn

class CustomClassificationHead(nn.Module):
    """Custom classification head with configurable architecture."""
    def __init__(self, hidden_size, num_classes, dropout_rate, num_hidden_layers=1, hidden_dim=None, activation='ReLU'):
        """ Initializes the custom classification head."""
        super().__init__()

        layers = []
        
        # First layer
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_size, hidden_dim))
        layers.append(getattr(nn, activation)())
        
        # Additional hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
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
    def __init__(self, base_model, custom_head):
        super().__init__()
        self.pre_trained_model = base_model
        self.classifier = custom_head
        
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
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        outputs = self.pre_trained_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Use [CLS] token representation
        cls_output = sequence_output[:, 0]
        logits = self.classifier(cls_output)

        return logits
