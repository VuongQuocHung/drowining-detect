"""
FastAPI Main Application

This is the main FastAPI application file that sets up the app,
includes routers, and handles startup events.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .models_loader import initialize_models, warm_up_models
from . import routes

# Set TensorFlow legacy mode
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Create FastAPI application
app = FastAPI(
    title="Drowning Detection System API",
    description="FastAPI backend for real-time drowning detection with YOLO and GAN enhancement",
    version="1.0.0"
)

# Configure CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Include routers
app.include_router(routes.router)


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    print("🚀 FastAPI backend starting up...")
    
    # Setup GPU
    # TensorFlow is initialized lazily when enhancement is requested.

    
    # Initialize models
    print("📊 Loading models...")
    yolo_models, deepsort_model, gan_model = initialize_models()
    
    # Store models in app state for access in routes
    app.state.models = yolo_models
    app.state.deepsort = deepsort_model
    app.state.gan_model = gan_model
    
    # Warm up models
    warm_up_models(yolo_models)
    
    # Log startup information
    print(f"📊 YOLO models loaded: {len(yolo_models)}")
    print(f"🎨 GAN model status: {'✅ Loaded' if gan_model else '❌ Not available'}")
    print("✨ Backend ready!")


@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Welcome to the FastAPI backend!"}


@app.on_event("shutdown")
def shutdown_event():
    from .sessions import sessions
    sessions.close()
