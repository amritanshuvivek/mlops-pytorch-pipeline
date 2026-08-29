import pytest
import torch
from src.model import get_model

def test_model_creation():
    """Test if ResNet-18 model is successfully created with 10 classes."""
    model = get_model(architecture="resnet18", num_classes=10)
    assert model is not None
    assert isinstance(model, torch.nn.Module)

def test_model_output_shape():
    """Test model outputs correct classification logits size for batch input."""
    model = get_model(architecture="resnet18", num_classes=10)
    model.eval()
    
    # CIFAR-10 image tensor shape: (batch_size, 3, 32, 32)
    dummy_input = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)
        
    assert output.shape == (2, 10)

def test_invalid_architecture():
    """Verify that an unsupported architecture raises ValueError."""
    with pytest.raises(ValueError):
        get_model(architecture="unsupported_cnn", num_classes=10)
