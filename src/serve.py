import io
import os
import sys
import torch
import yaml
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from src.model import get_model
from torchvision import transforms

app = FastAPI(title="CIFAR-10 Model Serving API")

# Global model container and device selection
model = None
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

@app.on_event("startup")
def load_model():
    global model
    config_path = os.getenv("CONFIG_PATH", "configs/training_config.yaml")
    
    if not os.path.exists(config_path):
        print(f"Startup error: Config file not found at {config_path}", file=sys.stderr)
        # We raise a RuntimeError so that the FastAPI app fails to start
        raise RuntimeError(f"Config file not found at {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", config["output"]["checkpoint_dir"])
    model_name = config["output"]["model_name"]
    checkpoint_path = os.path.join(checkpoint_dir, model_name)
    
    if not os.path.exists(checkpoint_path):
        print(f"Startup error: Checkpoint file not found at {checkpoint_path}", file=sys.stderr)
        raise RuntimeError(f"Checkpoint file not found at {checkpoint_path}")
        
    # Instantiate model structure
    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"]
    )
    
    # Load state dict
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        print(f"Model loaded successfully on {device} from {checkpoint_path}", flush=True)
    except Exception as e:
        print(f"Startup error loading checkpoint: {str(e)}", file=sys.stderr)
        raise RuntimeError(f"Error loading checkpoint: {str(e)}")

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or initialization failed")
    return {"status": "healthy"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or initialization failed")
        
    # Validate file extension
    filename = image.filename.lower() if image.filename else ""
    if not (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
        raise HTTPException(status_code=400, detail="Only JPEG or PNG images are supported.")
        
    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image content: {str(e)}")
        
    # Inference pre-processing (must match evaluation preprocessing: resize to 32x32, CIFAR-10 normalization)
    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    tensor_img = val_transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = torch.softmax(outputs, dim=1)[0].tolist()
        predicted_class_idx = torch.argmax(outputs, dim=1).item()
        
    return {
        "predicted_class": predicted_class_idx,
        "predicted_label": classes[predicted_class_idx],
        "probabilities": [round(p, 6) for p in probabilities]
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
