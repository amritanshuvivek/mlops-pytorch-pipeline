import torch
import torch.nn as nn
import torchvision.models as models

def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """
    Creates and returns a PyTorch model based on the architecture and number of classes.
    For CIFAR-10 (32x32 images), we adapt standard ResNet architectures to prevent 
    immediate aggressive downsampling and improve classification accuracy.
    """
    if architecture.lower() == "resnet18":
        # We load a standard resnet18 without weights
        model = models.resnet18(weights=None)
        
        # Modify the first conv layer (originally 7x7 stride 2, padding 3)
        # for CIFAR-10's 32x32 size to 3x3 stride 1, padding 1
        model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        
        # Replace the maxpool layer with Identity to preserve resolution in early layers
        model.maxpool = nn.Identity()
        
        # Modify the final linear classifier layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
