from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_dataloaders(
    data_dir: str = "/app/data",
    batch_size: int = 64,
    num_workers: int = 2
) -> tuple[DataLoader, DataLoader]:
    """
    Downloads and pre-processes the CIFAR-10 dataset.
    Returns:
        train_loader (DataLoader): DataLoader for CIFAR-10 training dataset.
        val_loader (DataLoader): DataLoader for CIFAR-10 test/validation dataset.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Standard CIFAR-10 mean and standard deviation for normalization
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)
    
    # Training transforms with simple data augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
    
    # Validation/Test transforms (only normalisation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
    
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def get_smoke_dataloaders(batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    """
    Creates dummy datasets using torchvision.datasets.FakeData for fast smoke testing.
    """
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
    
    # Use FakeData to simulate 3x32x32 images
    train_dataset = datasets.FakeData(
        size=256,
        image_size=(3, 32, 32),
        num_classes=10,
        transform=transform
    )
    
    val_dataset = datasets.FakeData(
        size=64,
        image_size=(3, 32, 32),
        num_classes=10,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
