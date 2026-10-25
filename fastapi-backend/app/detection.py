"""Detection adapters. Temporal alarm decisions belong to each session."""
import time
import cv2
import numpy as np
import torch
from .settings import ROOT, OBJECT_CLASSES, settings, normalize_class
from utils.data_utils import preprocess, deprocess

realtime_metrics = {"processing_speed_ms": 0., "real_time_speed_fps": 0., "latency_per_frame_ms": 0., "last_updated": 0.}
CLASS_COLORS = {"swimming": (0, 200, 0), "struggling": (0, 200, 255), "drowning": (0, 0, 255),
                "float": (255, 160, 0), "out_of_water": (200, 0, 200), "unknown": (128, 128, 128)}

def play_alarm_sound():
    if not settings.backend_alarm:
        return
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(str(ROOT / "sound/alarm.mp3"))
        pygame.mixer.music.play()
    except Exception as exc:
        print(f"Backend alarm unavailable: {exc}")

def check_drowning_and_trigger_alarm(detections, frame):
    """Compatibility helper: still images never update temporal alert state."""
    return frame

def run_detection(model, frame, conf_thresh=settings.confidence):
    with torch.inference_mode():
        results = model(frame, conf=conf_thresh, iou=0.4, agnostic_nms=True, verbose=False)
    detections = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                coords = tuple(map(int, box.xyxy[0].cpu()))
                label = normalize_class(result.names[int(box.cls[0].cpu())])
                detections.append((*coords, label, float(box.conf[0].cpu())))
    return detections

def draw_detections(frame, detections):
    for x1, y1, x2, y2, label, confidence in detections:
        color = CLASS_COLORS.get(label, CLASS_COLORS["unknown"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, max(18, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
    return frame

def track_with_deepsort(model, frame, deepsort, conf_thresh=settings.confidence, min_area=settings.min_area, is_video=True, timings=None):
    started = time.perf_counter()
    raw = run_detection(model, frame, conf_thresh)
    if timings is not None:
        timings["yolo_ms"] = (time.perf_counter()-started)*1000
    started = time.perf_counter()
    raw = [b for b in raw if (b[2]-b[0])*(b[3]-b[1]) >= min_area]
    objects = [b for b in raw if b[4] in OBJECT_CLASSES]
    raw = [b for b in raw if b[4] not in OBJECT_CLASSES]
    labels = ["swimming", "struggling", "drowning", "unknown"]
    tracked = []
    if not is_video:
        tracked = [dict(x1=b[0], y1=b[1], x2=b[2], y2=b[3], **{"class": b[4]}, confidence=b[5], class_id=labels.index(b[4]), track_id=i+1) for i,b in enumerate(raw)]
    else:
        xywh = [[(b[0]+b[2])/2, (b[1]+b[3])/2, b[2]-b[0], b[3]-b[1]] for b in raw]
        # Always predict/update, including empty frames, so tracks actually age.
        deepsort.update(torch.tensor(xywh, dtype=torch.float32).reshape(-1,4),
                        torch.tensor([b[5] for b in raw]), [labels.index(b[4]) for b in raw], frame)
        for track in deepsort.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update != 0:
                continue
            x1,y1,x2,y2 = deepsort._tlwh_to_xyxy(track.to_tlwh())
            tracked.append(dict(x1=x1,y1=y1,x2=x2,y2=y2,track_id=int(track.track_id),
                                class_id=int(track.oid),confidence=track.confidence, **{"class": labels[int(track.oid)]}))
    if timings is not None:
        timings["tracking_ms"] = (time.perf_counter()-started)*1000
    draw_detections(frame, objects)
    for det in tracked:
        label = det["class"]
        color = CLASS_COLORS[label]
        cv2.rectangle(frame,(det["x1"],det["y1"]),(det["x2"],det["y2"]),color,2)
        cv2.putText(frame, f"ID:{det['track_id']} {label} {det['confidence']:.2f}",
                    (det["x1"],max(18,det["y1"]-5)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
    return frame, tracked

def detection_with_enhance(frame, model, gan_model, enhancement=False, conf_thresh=settings.confidence):
    if enhancement and gan_model is not None:
        frame, _ = enhance_frame(frame, gan_model)
    detections = run_detection(model, frame, conf_thresh)
    return draw_detections(frame, detections), detections

def enhance_frame(frame, gan_model, side_by_side=False):
    """Enhance a single frame using GAN model for underwater image enhancement."""
    global realtime_metrics
    
    if gan_model is None:
        return frame, False
    
    try:
        start_time = time.time()
        
        # Convert BGR->RGB and resize to model's expected input (256x256)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_small = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)

        # Normalize [-1,1], predict, deprocess back to [0,255] uint8 RGB
        im = preprocess(rgb_small)
        im = np.expand_dims(im, axis=0)
        
        # Time the model prediction
        prediction_start = time.time()
        gen = gan_model.predict(im, verbose=0)
        prediction_end = time.time()
        
        gen_rgb_small = deprocess(gen)[0]  # (256,256,3) uint8

        # Upscale enhanced back to original frame size
        gen_rgb = cv2.resize(gen_rgb_small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
        gen_bgr = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2BGR)
        
        # Calculate metrics
        end_time = time.time()
        total_processing_time = end_time - start_time
        prediction_time = prediction_end - prediction_start
        
        # Update global metrics
        realtime_metrics["processing_speed_ms"] = round(prediction_time * 1000, 2)
        realtime_metrics["real_time_speed_fps"] = round(1.0 / total_processing_time if total_processing_time > 0 else 0, 2)
        realtime_metrics["latency_per_frame_ms"] = round(total_processing_time * 1000, 2)
        realtime_metrics["last_updated"] = time.time()

        if side_by_side:
            out = np.hstack((frame, gen_bgr))
            return out, True
        else:
            return gen_bgr, True

    except Exception as e:
        print(f"❌ Error enhancing frame: {e}")
        return frame, False


def enhance_video_frames_batch(frames, gan_model):
    """Process frames one by one to avoid GPU memory issues."""
    if gan_model is None or not frames:
        return [(frame, False) for frame in frames], False
    
    results = []
    for frame in frames:
        try:
            enhanced_frame, success = enhance_frame(frame, gan_model, side_by_side=False)
            results.append((enhanced_frame, success))
        except Exception as e:
            print(f"❌ Error enhancing frame: {e}")
            results.append((frame, False))
    
    return results, True


def get_realtime_metrics():
    """Get current real-time processing metrics."""
    global realtime_metrics
    
    # Check if metrics are recent (within last 5 seconds)
    current_time = time.time()
    metrics_age = current_time - realtime_metrics["last_updated"]
    
    if metrics_age > 5.0:  # If metrics are older than 5 seconds, return default values
        return {
            "processing_speed_ms": 0.0,
            "real_time_speed_fps": 0.0,
            "latency_per_frame_ms": 0.0,
            "metrics_available": False,
            "message": "No recent metrics available"
        }
    
    return {
        "processing_speed_ms": realtime_metrics["processing_speed_ms"],
        "real_time_speed_fps": realtime_metrics["real_time_speed_fps"],
        "latency_per_frame_ms": realtime_metrics["latency_per_frame_ms"],
        "metrics_available": True,
        "last_updated": realtime_metrics["last_updated"]
    }
