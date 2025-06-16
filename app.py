# Fixed FastAPI code
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import torch
from torchvision import transforms
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import uuid
from pathlib import Path

from model import RSGNet
from torchcam.methods import GradCAM
from torchvision.transforms.functional import to_pil_image

app = FastAPI()

# Create output directory if it doesn't exist
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RSGNet()

# Add error handling for model loading
try:
    model.load_state_dict(torch.load("rsgnet_weights.pth", map_location=device))
    print("Model weights loaded successfully")
except FileNotFoundError:
    print("Warning: Model weights file not found. Using random weights.")
except Exception as e:
    print(f"Error loading model weights: {e}")

model.to(device)
model.eval()

# Grad-CAM setup (targeting conv4)
cam_extractor = GradCAM(model, target_layer=model.conv4)

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Added normalization
])

# Store generated files temporarily
generated_files = {}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400, 
                content={"error": "File must be an image"}
            )
        
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")

        # Preprocess and prepare input
        tensor_image = preprocess(image).unsqueeze(0).to(device)
        tensor_image.requires_grad_()  # Needed for Grad-CAM

        # Forward pass WITH gradients enabled for GradCAM
        output = model(tensor_image)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item()

        # Compute Grad-CAM (this needs gradients)
        activation_maps = cam_extractor(predicted_class, output)
        activation_map = activation_maps[0].cpu().detach().numpy()
        
        # Remove batch dimension and resize to match image size
        if activation_map.ndim == 3:
            activation_map = activation_map.squeeze(0)  # Remove batch dimension
        
        # Resize activation map to match original image size (224x224) using PIL
        if activation_map.shape != (224, 224):
            # Convert to PIL Image, resize, then back to numpy
            activation_pil = Image.fromarray((activation_map * 255).astype(np.uint8))
            activation_pil = activation_pil.resize((224, 224), Image.BILINEAR)
            activation_map = np.array(activation_pil) / 255.0

        # Convert tensor back to PIL for visualization (without normalization)
        # Create a copy for visualization without normalization
        vis_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        vis_tensor = vis_preprocess(image).unsqueeze(0)
        original_image = to_pil_image(vis_tensor.squeeze())

        # Create visualization
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        axs[0].imshow(original_image)
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        axs[1].imshow(original_image)
        axs[1].imshow(activation_map, cmap='jet', alpha=0.4)
        axs[1].set_title(f"Grad-CAM (Class: {predicted_class})")
        axs[1].axis("off")

        # Generate unique filename
        file_id = str(uuid.uuid4())
        output_path = OUTPUT_DIR / f"gradcam_{file_id}.png"
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()  # Important: close the figure to free memory

        # Store file reference
        generated_files[file_id] = output_path

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "heatmap_id": file_id,
            "heatmap_url": f"/heatmap/{file_id}"
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed: {str(e)}"}
        )

@app.get("/heatmap/{file_id}")
def get_heatmap(file_id: str):
    if file_id not in generated_files:
        return JSONResponse(
            status_code=404,
            content={"error": "Heatmap not found"}
        )
    
    file_path = generated_files[file_id]
    if not file_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "Heatmap file not found"}
        )
    
    return FileResponse(
        file_path, 
        media_type="image/png",
        filename=f"gradcam_{file_id}.png"
    )

@app.on_event("startup")
async def startup_event():
    print(f"Model loaded on device: {device}")
    print(f"Output directory: {OUTPUT_DIR}")

@app.on_event("shutdown")
async def shutdown_event():
    # Cleanup generated files
    for file_path in generated_files.values():
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            print(f"Error cleaning up file {file_path}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)