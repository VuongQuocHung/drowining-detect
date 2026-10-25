"""
FastAPI Routes

This module contains all API endpoints for the drowning detection system.
"""

import os
import uuid
import base64
import time
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List

from .detection import (
    detection_with_enhance, enhance_frame
)
from .utils import (
    save_temp_file, calculate_visibility_metrics, create_analysis_result
)
from .gpu_utils import get_cuda_status
from .sessions import sessions, BusyError, inference_lock
from .models_loader import ensure_gan
from .settings import settings, ROOT, OBJECT_CLASSES


# Create router
router = APIRouter()

# Data models
class LoginRequest(BaseModel):
    """Pydantic model for user login requests."""
    email: str
    password: str

class User(BaseModel):
    """Pydantic model for user data."""
    id: int
    email: str
    name: str
    role: str
    certifications: List[str]
    shift: str
    status: str
    avatar: str

# Mock user database
users_db = [
    {
        "id": 1,
        "email": "lifeguard@admin.com",
        "password": "password123",
        "name": "Lifeguard / Pool Admin",
        "role": "Administrator",
        "certifications": ["Management"],
        "shift": "Full Day (8:00 AM - 6:00 PM)",
        "status": "on-duty",
        "avatar": "PA"
    }
]


# =============================================================================
# AUTHENTICATION ENDPOINTS
# =============================================================================

@router.post("/login")
def login(request: LoginRequest):
    """Authenticate user login credentials."""
    user = next((u for u in users_db if u["email"] == request.email and u["password"] == request.password), None)
    if user:
        user_data = user.copy()
        user_data.pop("password")
        return {"success": True, "token": "fake-jwt-token", "user": user_data}
    else:
        return {"success": False, "message": "Invalid email or password. Try Again!"}


# =============================================================================
# WEBCAM CONTROL ENDPOINTS
# =============================================================================

def session_or_404(session_id):
    session = sessions.items.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found or expired")
    return session


def validate_options(request, model_type, confidence, enhancement):
    if model_type not in request.app.state.models:
        raise HTTPException(400, "Invalid model_type")
    if not 0 <= confidence <= 1:
        raise HTTPException(400, "Confidence must be between 0 and 1")
    if enhancement and model_type != "underwater":
        raise HTTPException(400, "Enhancement is only supported underwater")


def create_session(request, model_type, source, enhancement, confidence):
    validate_options(request, model_type, confidence, enhancement)
    try:
        session = sessions.create(request.app, model_type, source, enhancement, confidence)
    except BusyError as exc:
        raise HTTPException(409, str(exc)) from exc
    return session


@router.post("/webcam/start")
def start_webcam(request: Request, model_type: str = Form("underwater"), enhancement: bool = Form(False), confidence_threshold: float = Form(settings.confidence)):
    session = create_session(request, model_type, 0, enhancement, confidence_threshold)
    return {"success": True, "message": "Camera session starting", "session_id": session.id,
            "stream_url": "/video_feed", "events_url": f"/sessions/{session.id}/events"}


@router.post("/webcam/stop")
def stop_webcam():
    session = sessions.active()
    if session and session.camera:
        session.stop()
    return {"success": True, "message": "Camera stopped"}


@router.get("/webcam/status")
def get_webcam_status():
    session = sessions.active()
    active = bool(session and session.camera)
    return {"active": active, "session_id": session.id if active else None,
            "model": session.model_type if active else None, "video_feed_active": active,
            "enhancement_active": session.enhancement if active else False,
            "confidence_threshold": session.confidence if active else None}


@router.post("/webcam/set_confidence")
def set_confidence_threshold(confidence_threshold: float = Form(...)):
    if not 0 <= confidence_threshold <= 1:
        raise HTTPException(400, "Confidence must be between 0 and 1")
    session = sessions.active()
    if session:
        raise HTTPException(409, "Stop and restart the source to change confidence")
    return {"success": True, "confidence_threshold": confidence_threshold,
            "message": "Pass confidence_threshold when starting the next source"}


@router.get("/sessions/active")
def active_session():
    session = sessions.active()
    return session.snapshot() if session else None


@router.get("/sessions/{session_id}/events")
def session_events(request: Request, session_id: str, after: int = 0):
    session = session_or_404(session_id)
    try:
        after = int(request.headers.get("last-event-id", after))
    except ValueError:
        raise HTTPException(400, "Invalid event cursor")
    return StreamingResponse(session.sse(after), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/realtime_metrics")
def get_realtime_metrics_endpoint():
    session = sessions.active()
    if session and session.events:
        metrics = session.events[-1].get("metrics", {})
        return {**metrics, "processing_speed_ms": metrics.get("yolo_ms", 0),
                "real_time_speed_fps": metrics.get("fps", 0), "latency_per_frame_ms": metrics.get("processing_ms", 0),
                "metrics_available": bool(metrics)}
    return {"metrics_available": False, "processing_speed_ms": 0, "real_time_speed_fps": 0, "latency_per_frame_ms": 0}


@router.get("/cuda_status")
def cuda_status():
    return get_cuda_status()


@router.get("/gan_model/status")
def gan_status(request: Request):
    from .settings import ROOT
    import importlib.util
    present = all((ROOT / "model/FUnIE_GAN_model" / name).exists() for name in ("model_15320_.h5", "model_15320_.json"))
    return {"available": present and importlib.util.find_spec("tensorflow") is not None,
            "loaded": request.app.state.gan_model is not None, "model_type": "FUnIE-GAN"}


@router.get("/video_feed")
def video_feed():
    session = sessions.active()
    if not session or not session.camera:
        raise HTTPException(400, "Webcam not active")
    return StreamingResponse(session.mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/live_camera_comparison")
def live_camera_comparison():
    session = sessions.active()
    if not session or not session.camera or not session.enhancement:
        raise HTTPException(400, "Start an enhanced webcam session first")
    return StreamingResponse(session.mjpeg(comparison=True), media_type="multipart/x-mixed-replace; boundary=frame")


@router.post("/predict")
def predict(request: Request, model_type: str = Form(...), source: str = Form(...), file: UploadFile = File(None), enhancement: bool = Form(False), confidence_threshold: float = Form(settings.confidence)):
    with sessions.lock:
        return predict_image(request, model_type, source, file, enhancement, confidence_threshold)


def predict_image(request, model_type, source, file, enhancement, confidence_threshold):
    validate_options(request, model_type, confidence_threshold, enhancement)
    if source != "image" or file is None:
        raise HTTPException(400, "Use source=image with a file; upload videos to /realtime_video_process")
    if sessions.active():
        raise HTTPException(409, "Stop the active source before analyzing an image")
    temp_path = save_temp_file(file)
    try:
        image = cv2.imread(temp_path)
        if image is None:
            raise HTTPException(400, "Invalid image")
        started = time.perf_counter()
        with inference_lock:
            gan = ensure_gan(request.app) if enhancement else None
            if enhancement and gan is None:
                raise HTTPException(503, "Enhancement model unavailable")
            frame, detections = detection_with_enhance(image, request.app.state.models[model_type], gan,
                                                       enhancement, confidence_threshold)
        labels = ["swimming", "struggling", "drowning", "unknown"]
        people = [b for b in detections if b[4] not in OBJECT_CLASSES]  # objects stay drawn on the image only
        tracked = [{"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "track_id": i+1,
                    "class": b[4], "class_id": labels.index(b[4]), "confidence": b[5]} for i,b in enumerate(people)]
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            raise HTTPException(500, "Cannot encode result")
        analysis = create_analysis_result(tracked, model_type)
        analysis["processingTime"] = f"{time.perf_counter()-started:.3f}s"
        return {"type": "image", "detections": tracked, "image_base64": base64.b64encode(buffer).decode(), "analysis": analysis}
    finally:
        os.remove(temp_path)


@router.post("/realtime_video_process")
def upload_video_for_realtime_processing(request: Request, model_type: str = Form(...), file: UploadFile = File(...), enhancement: bool = Form(False), confidence_threshold: float = Form(settings.confidence)):
    validate_options(request, model_type, confidence_threshold, enhancement)
    temp_path = save_temp_file(file)
    try:
        session = create_session(request, model_type, temp_path, enhancement, confidence_threshold)
    except Exception:
        os.remove(temp_path)
        raise
    return {"session_id": session.id, "stream_url": f"/realtime_video_stream/{session.id}",
            "events_url": f"/sessions/{session.id}/events"}


@router.get("/realtime_video_stream/{session_id}")
def realtime_video_stream(session_id: str):
    return StreamingResponse(session_or_404(session_id).mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/video_session_status/{session_id}")
def get_video_session_status(session_id: str):
    return session_or_404(session_id).snapshot()


@router.get("/download_processed_video/{session_id}")
def download_processed_video(session_id: str):
    session = session_or_404(session_id)
    if not session.snapshot()["download_available"]:
        raise HTTPException(400, "Processed video is not available")
    return FileResponse(str(session.output_path), filename=f"processed_detection_{session_id}.mp4", media_type="video/mp4")


@router.post("/stop_video_processing/{session_id}")
def stop_video_processing(session_id: str):
    session = session_or_404(session_id)
    session.stop()
    return {"success": True, "message": "Session stopped", **session.snapshot()}


# =============================================================================
# IMAGE AND VIDEO ENHANCEMENT ENDPOINTS
# =============================================================================

@router.post("/enhance_image")
def enhance_image(request: Request, file: UploadFile = File(...)):
    with sessions.lock:
        if any(s.status in ("starting", "running") for s in sessions.items.values()):
            raise HTTPException(409, "Stop the active source first")
        temp_path = save_temp_file(file)
        artifacts = []
        try:
            image = cv2.imread(temp_path)
            if image is None:
                raise HTTPException(400, "Invalid image")
            started = time.perf_counter()
            with inference_lock:
                gan = ensure_gan(request.app)
                if gan is None:
                    raise HTTPException(503, "Enhancement model unavailable")
                enhanced, success = enhance_frame(image, gan)
            if not success:
                raise HTTPException(500, "Enhancement failed")
            sid = uuid.uuid4().hex
            directory = ROOT / "processed_outputs"
            directory.mkdir(parents=True,exist_ok=True)
            for kind, frame in (("original",image),("enhanced",enhanced),("comparison",np.hstack((image,enhanced)))):
                path = directory / f"{sid}_{kind}.jpg"
                if not cv2.imwrite(str(path),frame):
                    raise HTTPException(500, "Cannot save enhanced image")
                artifacts.append(path)
            sessions.register_artifacts(artifacts)
            metrics = calculate_visibility_metrics(image, enhanced)
            elapsed = (time.perf_counter()-started)*1000
            return {"success": True, "session_id":sid, "message":"Image enhanced",
                    "original_size":{"width":image.shape[1],"height":image.shape[0]},
                    **{f"{kind}_url":f"/download_enhanced_image/{sid}/{kind}" for kind in ("original","enhanced","comparison")},
                    "metrics":{**metrics, "visibility_gain":metrics.get("visibility_gain_percentage",0),
                               "processing_time_ms":round(elapsed,2), "total_latency_ms":round(elapsed,2),
                               "real_time_speed_fps":round(1000/max(elapsed,.001),2)}}
        except Exception:
            for path in artifacts:
                path.unlink(missing_ok=True)
            raise
        finally:
            os.remove(temp_path)


@router.get("/download_enhanced_image/{session_id}/{image_type}")
def download_enhanced_image(session_id: str, image_type: str):
    if image_type not in ("original","enhanced","comparison") or not session_id.isalnum():
        raise HTTPException(400, "Invalid image type or session")
    path = ROOT / "processed_outputs" / f"{session_id}_{image_type}.jpg"
    if not path.is_file():
        raise HTTPException(404, "Image not found or expired")
    return FileResponse(str(path),media_type="image/jpeg",filename=path.name)


@router.post("/enhance_video")
def enhance_video(request: Request, file: UploadFile = File(...), create_comparison: bool = Form(False), real_time: bool = Form(True)):
    temp_path = save_temp_file(file)
    try:
        session = sessions.create(request.app, "underwater", temp_path, True, settings.confidence,
                                  detection=False, comparison=create_comparison)
    except BusyError as exc:
        os.remove(temp_path)
        raise HTTPException(409,str(exc)) from exc
    except Exception:
        os.remove(temp_path)
        raise
    if not real_time:
        session.thread.join()
        if session.status != "completed":
            raise HTTPException(500,session.error or "Enhancement interrupted")
    return {"success":True, "session_id":session.id,"original_filename":file.filename,
            "real_time":real_time, "frame_count":session.frame_count,
            "processing_time":f"{time.monotonic()-session.created:.2f}s",
            "enhanced_video_path":f"/download_enhanced_video/{session.id}",
            "comparison_video_path":f"/download_enhanced_video/{session.id}?video_type=comparison" if create_comparison else None,
            "stream_url":f"/enhance_video_stream/{session.id}",
            "stream_urls":{"comparison":f"/enhance_video_stream/{session.id}","enhanced_only":f"/enhance_video_stream_only/{session.id}"},
            "message":"Enhancement session started" if real_time else "Enhancement completed"}


@router.get("/enhance_video_stream/{session_id}")
def enhance_video_stream(session_id: str):
    return StreamingResponse(session_or_404(session_id).mjpeg(comparison=True),media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/enhance_video_stream_only/{session_id}")
def enhance_video_stream_only(session_id: str):
    return StreamingResponse(session_or_404(session_id).mjpeg(),media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/download_enhanced_video/{session_id}")
def download_enhanced_video(session_id: str, video_type: str = "enhanced"):
    session = session_or_404(session_id)
    if video_type not in ("enhanced","comparison"):
        raise HTTPException(400,"Invalid video_type")
    path = session.output_path if video_type == "enhanced" else session.comparison_path
    if session.status != "completed" or not path.is_file():
        raise HTTPException(404,"Video not completed or expired")
    return FileResponse(str(path),media_type="video/mp4",filename=f"{session_id}_{video_type}.mp4")
