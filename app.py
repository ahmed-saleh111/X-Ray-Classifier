import torch
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from vit import predict_image as vit_predict
from transformers import ViTForImageClassification, ViTImageProcessor
import uvicorn
from pathlib import Path
import shutil
from typing import Dict, Any

app = FastAPI(title="X-ray Pneumonia Prediction API", version="1.0.0")

UPLOAD_FOLDER = "static/uploads"
TEMPLATES_DIR = "templates"

Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(TEMPLATES_DIR).mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ViTForImageClassification.from_pretrained('models')
processor = ViTImageProcessor.from_pretrained('models')
model.to(device)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_vit(image: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict pneumonia from X-ray image
    
    Args:
        image: Uploaded image file
        
    Returns:
        Dictionary containing diagnosis and confidence
    """

    if not image.filename:
        raise HTTPException(status_code=400, detail="No image selected")
    
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        file_extension = Path(image.filename).suffix
        safe_filename = f"temp_image{file_extension}"
        file_path = Path(UPLOAD_FOLDER) / safe_filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        with open(file_path, 'rb') as img_file:
            result = vit_predict(model, processor, img_file, device)
            print(f"Prediction result: {result}")
        
        file_path.unlink(missing_ok=True)
        
        return result
        
    except Exception as e:
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": device,
        "model_loaded": model is not None
    }

if __name__ == '__main__':
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5002,
        reload=True,
        log_level="info"
    )