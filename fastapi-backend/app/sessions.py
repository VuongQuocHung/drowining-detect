"""One processing worker per source; MJPEG/SSE only subscribe to its results."""
import json
import logging
import os
import queue
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from .alerts import AlertEngine
from .settings import ROOT, settings

log = logging.getLogger(__name__)
inference_lock = threading.Lock()

class BusyError(Exception):
    pass

class Session:
    def __init__(self, app, model_type, source, enhancement, confidence, detection=True, comparison=False, frame_limit=None):
        self.id = uuid.uuid4().hex
        self.app, self.model_type, self.source = app, model_type, source
        self.camera = isinstance(source, int)
        self.enhancement, self.confidence = enhancement, confidence
        self.detection, self.comparison = detection, comparison
        self.frame_limit = frame_limit
        self.frame_count, self.fps, self.width, self.height = 0, 0, 0, 0
        self.alerts = AlertEngine()
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.thread = None
        self.status, self.error = "starting", None
        self.created = self.updated = time.monotonic()
        self.jpeg, self.comparison_jpeg, self.sequence = None, None, 0
        self.events = deque(maxlen=300)
        self.latencies = deque(maxlen=300)
        self.output_path = ROOT / "processed_outputs" / f"{self.id}.mp4"
        self.log_path = self.output_path.with_suffix(".jsonl")
        self.comparison_path = self.output_path.with_name(f"{self.id}_comparison.mp4")
        self.cap = None

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True, name=f"source-{self.id}")
        self.thread.start()

    def publish(self, payload):
        with self.condition:
            self.sequence += 1
            self.updated = time.monotonic()
            self.events.append({"sequence": self.sequence, "session_id": self.id,
                                "status": self.status, **payload})
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(self.events[-1], allow_nan=False) + "\n")
            except OSError:
                log.exception("Could not write session telemetry")
            self.condition.notify_all()

    def snapshot(self):
        return {"session_id": self.id, "model_type": self.model_type, "source": "webcam" if self.camera else "video",
                "status": self.status, "error": self.error,
                "mode": "detection" if self.detection else "enhancement", "frame_count": self.frame_count,
                "processing_complete": self.status == "completed",
                "download_available": not self.camera and self.status == "completed" and self.output_path.exists(),
                "enhancement": self.enhancement, "confidence_threshold": self.confidence}

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)

    def run(self):
        import cv2
        import numpy as np
        import psutil
        from .detection import enhance_frame, track_with_deepsort, play_alarm_sound
        from .models_loader import load_deepsort_model, ensure_gan
        writer, comparison_writer, reader = None, None, None
        latest = queue.Queue(maxsize=1)
        process = psutil.Process()
        process.cpu_percent()
        reader_done = threading.Event()
        outcome = "failed"
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise ValueError("Cannot open source")
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            fps = fps if np.isfinite(fps) and 0 < fps <= 240 else 30.0
            self.fps = fps
            with inference_lock:
                tracker = load_deepsort_model() if self.detection else None
                gan = ensure_gan(self.app) if self.enhancement else None
            if self.enhancement and gan is None:
                raise ValueError("Enhancement model unavailable")
            self.status = "running"

            def capture_latest():
                try:
                    while not self.stop_event.is_set():
                        started = time.perf_counter()
                        ok, frame = self.cap.read()
                        if not ok:
                            break
                        item = (frame, time.monotonic(), (time.perf_counter()-started)*1000)
                        try:
                            latest.get_nowait()
                        except queue.Empty:
                            pass
                        latest.put_nowait(item)
                finally:
                    reader_done.set()

            if self.camera:
                reader = threading.Thread(target=capture_latest, daemon=True)
                reader.start()
            count, origin, previous_timestamp = 0, time.monotonic(), -1.0
            while not self.stop_event.is_set():
                tick = time.perf_counter()
                timings = {"gan_ms": 0.0}
                if self.camera:
                    try:
                        frame, captured, timings["read_ms"] = latest.get(timeout=.25)
                    except queue.Empty:
                        if reader_done.is_set():
                            raise ValueError("Camera disconnected")
                        continue
                    timestamp = max(0.0, captured - origin)
                    tick = time.perf_counter()
                else:
                    ok, frame = self.cap.read()
                    timings["read_ms"] = (time.perf_counter()-tick)*1000
                    if not ok:
                        if count == 0:
                            raise ValueError("Video has no decodable frames")
                        break
                    timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    if not np.isfinite(timestamp) or timestamp <= previous_timestamp:
                        timestamp = count / fps
                    timestamp = max(timestamp, previous_timestamp)
                previous_timestamp = timestamp
                with inference_lock:
                    if self.enhancement:
                        original = frame.copy()
                        started = time.perf_counter()
                        frame, success = enhance_frame(frame, gan)
                        timings["gan_ms"] = (time.perf_counter()-started)*1000
                        if not success:
                            raise ValueError("Enhancement failed; disable it and restart")
                    if self.detection:
                        frame, detections = track_with_deepsort(self.app.state.models[self.model_type], frame, tracker,
                                                               conf_thresh=self.confidence, timings=timings)
                    else:
                        detections = []
                        timings.update(yolo_ms=0.0, tracking_ms=0.0)
                people = self.alerts.update(detections, timestamp)
                if any(p["alarm"] for p in people):
                    play_alarm_sound()
                if not self.camera:
                    self.width, self.height = frame.shape[1], frame.shape[0]
                    if writer is None:
                        self.output_path.parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(str(self.output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
                        if not writer.isOpened():
                            raise ValueError("Cannot create processed video")
                    writer.write(frame)
                    if self.comparison and self.enhancement:
                        if comparison_writer is None:
                            comparison_writer = cv2.VideoWriter(str(self.comparison_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1]*2,frame.shape[0]))
                            if not comparison_writer.isOpened():
                                raise ValueError("Cannot create comparison video")
                        comparison_writer.write(np.hstack((original,frame)))
                started = time.perf_counter()
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    raise ValueError("JPEG encoding failed")
                timings["jpeg_ms"] = (time.perf_counter()-started)*1000
                self.jpeg = encoded.tobytes()
                if self.enhancement:
                    ok, comparison = cv2.imencode(".jpg", np.hstack((original, frame)), [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        self.comparison_jpeg = comparison.tobytes()
                timings["jpeg_ms"] = (time.perf_counter()-started)*1000
                duration = (time.perf_counter()-tick)*1000
                latency = (time.monotonic()-captured)*1000 if self.camera else duration
                self.latencies.append(latency)
                count += 1
                self.frame_count = count
                metrics = {**{k: round(v, 2) for k,v in timings.items()},
                           "processing_ms": round(duration, 2), "latency_p50_ms": round(float(np.percentile(self.latencies,50)),2),
                           "latency_p95_ms": round(float(np.percentile(self.latencies,95)),2),
                           "fps": round(count/max(time.monotonic()-origin, .001),2),
                           "rss_mb": round(process.memory_info().rss/1024**2,2), "cpu_percent": process.cpu_percent()}
                self.publish({"source_time": round(timestamp,3), "detections": people, "metrics": metrics})
                if self.frame_limit is not None and count >= self.frame_limit:
                    break
                if not self.camera:
                    self.stop_event.wait(max(0, 1/fps-(time.perf_counter()-tick)))
            outcome = "stopped" if self.stop_event.is_set() else "completed"
        except Exception as exc:
            outcome, self.error = "failed", str(exc)
            log.exception("Session %s failed", self.id)
        finally:
            self.stop_event.set()
            if self.cap:
                self.cap.release()
            if reader:
                reader.join(timeout=2)
            if writer:
                writer.release()
            if comparison_writer:
                comparison_writer.release()
            try:
                if not self.camera:
                    Path(self.source).unlink(missing_ok=True)
                if outcome != "completed":
                    self.output_path.unlink(missing_ok=True)
                    self.comparison_path.unlink(missing_ok=True)
            except OSError as exc:
                outcome, self.error = "failed", f"Cleanup failed: {exc}"
                log.exception("Cleanup failed for %s", self.id)
            self.status = outcome
            self.publish({"error": self.error, "detections": [], "terminal": True})

    def mjpeg(self, comparison=False):
        sequence = -1
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.sequence != sequence or self.stop_event.is_set(), timeout=15)
                sequence, jpeg = self.sequence, self.comparison_jpeg if comparison else self.jpeg
                terminal = self.status in ("completed", "failed", "stopped")
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            if terminal:
                return

    def sse(self, after=0):
        # Reconnect replays buffered events; workers never depend on subscribers.
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.sequence > after, timeout=10)
                events = [e for e in self.events if e["sequence"] > after]
                if after == 0 and events:
                    # First subscription restores current state without replaying old sounds.
                    events = [{**events[-1], "detections": [{**p, "alarm": False} for p in events[-1].get("detections", [])]}]
                terminal = self.status in ("completed", "failed", "stopped")
            if not events:
                yield ": heartbeat\n\n"
            for event in events:
                after = event["sequence"]
                yield f"id: {after}\ndata: {json.dumps(event, allow_nan=False)}\n\n"
            if terminal:
                return

class SessionManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.items = {}
        self.closed = threading.Event()
        self.janitor = None
        self.artifacts = []

    def create(self, app, model_type, source, enhancement, confidence, detection=True, comparison=False, frame_limit=None):
        with self.lock:
            if any(s.status in ("starting", "running") or (s.thread and s.thread.is_alive()) for s in self.items.values()):
                raise BusyError("Stop the active source before starting another")
            session = Session(app, model_type, source, enhancement, confidence, detection, comparison, frame_limit)
            self.items[session.id] = session
            session.start()
            if self.janitor is None:
                self.janitor = threading.Thread(target=self.reap, daemon=True)
                self.janitor.start()
            return session

    def reap(self):
        while not self.closed.wait(30):
            with self.lock:
                for sid, session in list(self.items.items()):
                    if session.stop_event.is_set() and (not session.thread or not session.thread.is_alive()) and time.monotonic()-session.updated > settings.session_ttl:
                        session.output_path.unlink(missing_ok=True)
                        session.log_path.unlink(missing_ok=True)
                        session.comparison_path.unlink(missing_ok=True)
                        del self.items[sid]
                for created, paths in list(self.artifacts):
                    if time.monotonic()-created > settings.session_ttl:
                        for path in paths:
                            path.unlink(missing_ok=True)
                        self.artifacts.remove((created,paths))

    def register_artifacts(self, paths):
        # Caller holds self.lock while doing single-image inference.
        self.artifacts.append((time.monotonic(), paths))
        if self.janitor is None:
            self.janitor = threading.Thread(target=self.reap, daemon=True)
            self.janitor.start()

    def active(self):
        with self.lock:
            return next((s for s in self.items.values() if s.status in ("starting", "running")), None)

    def close(self):
        self.closed.set()
        for session in list(self.items.values()):
            session.stop()

sessions = SessionManager()
