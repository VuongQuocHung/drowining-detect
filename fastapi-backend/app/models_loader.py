"""
Model Loading and Initialization

This module handles loading and initialization of YOLO detection models
and FUnIE-GAN enhancement models.
"""

import os
import numpy as np
import torch
from ultralytics import YOLO
from .settings import ROOT, settings
from threading import Lock
_gan_lock = Lock()

from .gpu_utils import get_device, setup_pytorch_cuda
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort


def load_gan_model():
    """Load the FUnIE-GAN model for underwater video enhancement.
    
    Returns:
        tensorflow.keras.Model or None: Loaded GAN model or None if loading fails
    """
    try:
        from .gpu_utils import setup_gpu, optimize_tensorflow_for_inference
        setup_gpu()
        optimize_tensorflow_for_inference()
        from tensorflow.keras.models import model_from_json
        checkpoint_dir = str(ROOT / "model/FUnIE_GAN_model") + "/"
        model_name_by_epoch = "model_15320_"
        model_h5 = checkpoint_dir + model_name_by_epoch + ".h5"
        model_json = checkpoint_dir + model_name_by_epoch + ".json"
        
        if not (os.path.exists(model_h5) and os.path.exists(model_json)):
            print("⚠️ GAN model files not found. Video enhancement will be disabled.")
            return None
        
        with open(model_json, "r") as json_file:
            loaded_model_json = json_file.read()
        
        model = model_from_json(loaded_model_json)
        model.load_weights(model_h5)
        print("✅ Loaded FUnIE-GAN model successfully")
        
        # Warm-up on GPU (important for first-frame latency)
        dummy = np.zeros((1, 256, 256, 3), dtype=np.float32)
        _ = model.predict(dummy)
        return model
        
    except Exception as e:
        print(f"❌ Failed to load GAN model: {str(e)}")
        return None


def load_yolo_models():
    """Load YOLO detection models.
    
    Returns:
        dict: Dictionary containing loaded YOLO models
    """
    device = get_device()
    print(f"🔧 Using device: {device}")
    
    # Load YOLO Detection Models
    models = {
        "underwater": YOLO(settings.underwater_model).to(device),
        "above-water": YOLO(settings.above_model).to(device),
    }
    
    print(f"✅ YOLO models loaded on {device}")
    if torch.cuda.is_available():
        print(f"🚀 CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return models


def load_deepsort_model():
    """Load DeepSORT tracking model.
    
    Returns:
        DeepSort: Initialized DeepSORT tracker
    """
    deepsort_cfg_path = str(ROOT / "deep_sort_pytorch/configs/deep_sort.yaml")
    _cfg = get_config()
    _cfg.merge_from_file(deepsort_cfg_path)
    checkpoint = ROOT / _cfg.DEEPSORT.REID_CKPT
    if not checkpoint.is_file():
        raise FileNotFoundError("DeepSORT checkpoint missing. Run python tools/fetch_tracker.py from fastapi-backend.")
    
    deepsort = DeepSort(
        str(ROOT / _cfg.DEEPSORT.REID_CKPT),
        max_dist=_cfg.DEEPSORT.MAX_DIST,
        min_confidence=0.0,
        nms_max_overlap=_cfg.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=_cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=_cfg.DEEPSORT.MAX_AGE,
        n_init=_cfg.DEEPSORT.N_INIT,
        nn_budget=_cfg.DEEPSORT.NN_BUDGET,
        use_cuda=torch.cuda.is_available()
    )
    
    return deepsort


def initialize_models():
    """Initialize all models (YOLO, DeepSORT, and GAN).
    
    Returns:
        tuple: (yolo_models, deepsort_model, gan_model)
    """
    # Setup PyTorch CUDA optimizations
    setup_pytorch_cuda()
    
    # Load models
    yolo_models = load_yolo_models()
    deepsort_model = None
    gan_model = None
    
    return yolo_models, deepsort_model, gan_model


def warm_up_models(models):
    """Warm up models with dummy inference for better performance.
    
    Args:
        models: Dictionary of YOLO models
    """
    if torch.cuda.is_available():
        print("🔥 Warming up CUDA with dummy inference...")
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        for model_name, model in models.items():
            try:
                with torch.no_grad():
                    _ = model(dummy_frame, verbose=False)
                print(f"✅ {model_name} model warmed up on GPU")
            except Exception as e:
                print(f"⚠️ Warning: Could not warm up {model_name} model: {e}")


def ensure_gan(app):
    with _gan_lock:
        if app.state.gan_model is None:
            app.state.gan_model = load_gan_model()
        return app.state.gan_model
