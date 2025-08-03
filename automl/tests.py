import pandas as pd
import torch
from unittest.mock import MagicMock
from ray.tune import tune
from automl.bohb_core import TextAutoML


def test_overfit_loop_skips_training(tmp_path):
    automl = TextAutoML(
        normalized_class_weights=None,
        seed=42,
        token_length=128,
        max_epochs=3,
        batch_size=2,
        lr=1e-4,
        weight_decay=1e-5,
        train_df=pd.DataFrame({'text': ["a", "b"], 'label': [0, 1]}),
        val_df=pd.DataFrame({'text': ["c", "d"], 'label': [0, 1]})
    )

    automl.overfit = True
    automl.starting_epoch = 0
    automl.max_validation_accuracy = 0.99

    # Mock save_extra_info to track call
    automl.save_extra_info = MagicMock()
    tune.report = MagicMock()

    automl.overfit_loop()

    assert automl.save_extra_info.call_count == 3  # Should save info per epoch


def test_best_model_saved_before_overfit():
    # Step 1: Setup minimal training environment
    automl = TextAutoML(
        normalized_class_weights=None,
        seed=42,
        token_length=128,
        max_epochs=5,
        batch_size=2,
        lr=1e-4,
        weight_decay=1e-5,
        train_df=pd.DataFrame({'text': ["a", "b"], 'label': [0, 1]}),
        val_df=pd.DataFrame({'text': ["c", "d"], 'label': [0, 1]})
    )

    # Step 2: Mock the model and optimizer with dummy state
    automl.create_model(
        fraction_layers_to_finetune=2,
        num_classes=2,
        classification_head_hidden_dim=128,
        classification_head_dropout_rate=0.2,
        classification_head_hidden_layers=2,
        classification_head_activation="ReLU",
        use_layer_norm=True
    )
    dummy_model = MagicMock()
    dummy_optimizer = MagicMock()
    dummy_model.state_dict.side_effect = [
        {"weights": torch.tensor([0.6])},  # Epoch 1: best
        {"weights": torch.tensor([0.58])},  # Epoch 2
        {"weights": torch.tensor([0.56])},  # Epoch 3
        {"weights": torch.tensor([0.55])},  # Epoch 4 (early stop triggered)
    ]
    dummy_optimizer.state_dict.return_value = {"opt": torch.tensor([1.0])}

    dummy_model.load_state_dict = MagicMock()
    dummy_optimizer.load_state_dict = MagicMock()

    automl.model = dummy_model
    automl.optimizer = dummy_optimizer

    # Step 3: Fake validation accuracies per epoch
    val_accuracies = [0.6, 0.58, 0.56, 0.55]

    for epoch, acc in enumerate(val_accuracies):
        automl.starting_epoch = epoch
        if acc > automl.max_validation_accuracy:
            automl.max_validation_accuracy = acc
            automl.no_improvement_count = 0
            automl.save_best_model()
        else:
            automl.no_improvement_count += 1
            if automl.no_improvement_count >= 3:
                automl.overfit = True
                automl.load_best_model()
                break

    # Step 4: Check that overfitting was triggered
    assert automl.overfit is True

    # Step 5: Check that best model state was saved from epoch 0 (val_acc 0.6)
    assert automl.best_model_state == {"weights": torch.tensor([0.6])}

    # Step 6: Ensure best model was reloaded
    dummy_model.load_state_dict.assert_called_with({"weights": torch.tensor([0.6])})
    dummy_optimizer.load_state_dict.assert_called_with({"opt": torch.tensor([1.0])})
