import argparse
import json
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from src.model import get_model
from src.dataset import get_dataloaders, get_smoke_dataloaders

def log_event(event_name: str, **kwargs):
    """Utility to print a structured JSON event to stdout."""
    event = {"event": event_name}
    event.update(kwargs)
    print(json.dumps(event), flush=True)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on CIFAR-10")
    parser.add_argument(
        "--config", 
        type=str, 
        default=os.getenv("CONFIG_PATH", "configs/training_config.yaml"),
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run fast smoke test with mock data"
    )
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}", file=sys.stderr)
        sys.exit(1)
        
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Extract configs
    model_config = config["model"]
    train_config = config["training"]
    data_config = config["data"]
    output_config = config["output"]
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    log_event("training_init", device=str(device), config=config)
    
    # Create checkpoint directory (allow environment override for local run)
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", output_config["checkpoint_dir"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Get dataloaders (allow environment override or smoke test mode)
    if args.smoke_test or os.getenv("SMOKE_TEST") == "true":
        log_event("smoke_test_mode", message="Using mock data for fast verification")
        train_loader, val_loader = get_smoke_dataloaders(
            batch_size=train_config["batch_size"]
        )
    else:
        data_dir = os.getenv("DATA_DIR", data_config["data_dir"])
        train_loader, val_loader = get_dataloaders(
            data_dir=data_dir,
            batch_size=train_config["batch_size"],
            num_workers=2
        )
    
    # Instantiate model
    model = get_model(
        architecture=model_config["architecture"],
        num_classes=model_config["num_classes"]
    )
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_config["learning_rate"])
    
    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_no_improve = 0
    early_stopping_patience = train_config.get("early_stopping_patience", 3)
    checkpoint_path = os.path.join(checkpoint_dir, output_config["model_name"])
    
    log_event("training_start", total_epochs=train_config["epochs"])
    
    for epoch in range(1, train_config["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Metrics output as JSON Lines
        metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4)
        }
        print(json.dumps(metrics), flush=True)
        
        # Checkpoint and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_no_improve = 0
            
            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc
            }
            torch.save(checkpoint, checkpoint_path)
            log_event(
                "checkpoint_saved",
                epoch=epoch,
                val_loss=round(val_loss, 4),
                val_accuracy=round(val_acc, 4),
                path=checkpoint_path
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                log_event("early_stopping", epoch=epoch, patience=early_stopping_patience)
                break
                
    log_event(
        "training_complete",
        best_val_loss=round(best_val_loss, 4),
        best_val_accuracy=round(best_val_acc, 4),
        checkpoint_path=checkpoint_path
    )

if __name__ == "__main__":
    main()
