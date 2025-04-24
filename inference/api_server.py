import os
import torch
import yaml
import numpy as np
import logging
import time
import uuid
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import from inference_outfit.py
from inference_outfit import (
    generate_outfit_embedding_from_text, 
    setup_logging,
    load_average_outfit_embedding
)

# Import needed modules for model loading
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.OutfitGTN import OutfitGTN
from utils.gpu_utils import configure_gpu, get_device

# Load environment variables
load_dotenv()

# Set up logging
logger = setup_logging()

# API security setup
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    """Validate API key from request header"""
    expected_key = os.getenv("API_KEY")
    if not expected_key:
        logger.warning("No API key configured")
        return api_key_header
        
    if api_key_header != expected_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key_header

# Request and response models
class EmbeddingRequest(BaseModel):
    item_descriptions: List[str] = Field(
        ..., 
        description="List of item descriptions", 
        example=["Blue denim jeans", "White cotton t-shirt"]
    )
    outfit_description: Optional[str] = Field(
        None, 
        description="Optional outfit description", 
        example="Casual summer outfit"
    )

class EmbeddingResponse(BaseModel):
    request_id: str = Field(..., description="Unique request ID")
    embedding: List[float] = Field(..., description="Generated OutfitGTN embedding")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")

# Global model instance
model = None
device = None

# FastAPI app initialization
app = FastAPI(
    title="OutfitGTN Embedding API",
    description="API for generating OutfitGTN embeddings from text descriptions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the model on server startup"""
    global model, device
    
    logger.info("Initializing OutfitGTN model...")
    configure_gpu()
    device = get_device()
    
    # Load model configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config/config_GTN.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = OutfitGTN(
        input_dim=config['model']['input_dim'],
        hidden_channels=config['model']['hidden_channels'],
        embedding_dim=config['model']['embedding_dim'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        residual=config['model']['residual'],
        temperature=config['model']['temperature'],
        enable_drop=False,
        drop_rate=0.0
    )
    
    # Load weights
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints/best_model_GTN_2025-04-15_16-46-00.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully. Using device: {device}")
    
    # Verify average embedding is available
    try:
        _ = load_average_outfit_embedding()
        logger.info("Average outfit embedding loaded successfully")
    except Exception as e:
        logger.warning(f"Average outfit embedding not found: {e}")

@app.post("/generate-embedding", response_model=EmbeddingResponse)
async def generate_embedding(
    request: EmbeddingRequest,
    api_key: str = Security(get_api_key)
) -> EmbeddingResponse:
    """Generate OutfitGTN embedding from text descriptions"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    if len(request.item_descriptions) == 0:
        raise HTTPException(status_code=400, detail="At least one item description is required")
    
    request_id = str(uuid.uuid4())
    logger.info(f"Processing embedding request {request_id} with {len(request.item_descriptions)} items")
    
    start_time = time.time()
    
    try:
        # Generate embedding
        embedding = generate_outfit_embedding_from_text(
            model=model,
            device=device,
            item_descriptions=request.item_descriptions,
            outfit_description=request.outfit_description
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        processing_time_ms = int(processing_time * 1000)
        
        logger.info(f"Request {request_id} processed in {processing_time_ms}ms")
        
        # Convert embedding to list for JSON serialization
        embedding_list = embedding.tolist()
        
        return EmbeddingResponse(
            request_id=request_id,
            embedding=embedding_list,
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Basic health check endpoint"""
    return {"status": "healthy", "service": "OutfitGTN Embedding API"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8004"))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=False)