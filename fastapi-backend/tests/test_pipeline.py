import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app import routes
from app.sessions import SessionManager
from deep_sort_pytorch.deep_sort.deep_sort import DeepSort
from deep_sort_pytorch.deep_sort.sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort_pytorch.deep_sort.sort.tracker import Tracker

def tracker():
    # Exercise the actual association/Kalman lifecycle without requiring a ReID GPU.
    instance = DeepSort.__new__(DeepSort)
    instance.min_confidence = 0
    instance.extractor = lambda crops: np.tile(np.array([1., 0., 0.]), (len(crops),1))
    instance.tracker = Tracker(NearestNeighborDistanceMetric("cosine", .2, 100), n_init=3, max_age=3)
    return instance

class TrackingTest(unittest.TestCase):
    def test_label_updates_and_empty_frames_age_tracks(self):
        from app.detection import track_with_deepsort
        ds, frame = tracker(), np.zeros((120,120,3),dtype=np.uint8)
        with patch("app.detection.run_detection", return_value=[(20,20,80,100,"swimming",.9)]) as detect:
            for _ in range(3):
                _, people = track_with_deepsort(None,frame.copy(),ds)
            tid = people[0]["track_id"]
            detect.return_value = [(20,20,80,100,"drowning",.87)]
            _, people = track_with_deepsort(None,frame.copy(),ds)
            self.assertEqual(people[0]["track_id"],tid)
            self.assertEqual(people[0]["class"],"drowning")
            self.assertAlmostEqual(people[0]["confidence"],.87,places=5)
            detect.return_value = []
            for _ in range(4):
                _, people = track_with_deepsort(None,frame.copy(),ds)
                self.assertEqual(people,[])
            self.assertFalse(ds.tracker.tracks)

class ApiTest(unittest.TestCase):
    def setUp(self):
        self.manager = SessionManager()
        self.app = FastAPI()
        self.app.state.models = {"underwater": object(), "above-water": object()}
        self.app.state.gan_model = None
        self.app.include_router(routes.router)
        self.client = TestClient(self.app)
        self.patches = [patch.object(routes,"sessions",self.manager),
                        patch("app.models_loader.load_deepsort_model",side_effect=tracker),
                        patch("app.detection.run_detection",return_value=[(20,20,80,100,"drowning",.9)])]
        for item in self.patches:
            item.start()

    def tearDown(self):
        self.manager.close()
        for session in self.manager.items.values():
            session.output_path.unlink(missing_ok=True)
            session.log_path.unlink(missing_ok=True)
            session.comparison_path.unlink(missing_ok=True)
        for item in reversed(self.patches):
            item.stop()
        self.client.close()

    def video(self, frames=8):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"input.mp4"
            writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),20,(120,120))
            self.assertTrue(writer.isOpened())
            for _ in range(frames):
                writer.write(np.zeros((120,120,3),dtype=np.uint8))
            writer.release()
            return path.read_bytes()

    def upload(self, payload):
        return self.client.post("/realtime_video_process", data={"model_type":"underwater"}, files={"file":("test.mp4",payload,"video/mp4")})

    def finished(self, sid):
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            response = self.client.get(f"/video_session_status/{sid}").json()
            if response["status"] in ("completed","failed","stopped"):
                return response
            time.sleep(.02)
        self.fail("Worker did not finish")

    def test_single_source_stop_then_restart(self):
        payload = self.video(60)
        first = self.upload(payload)
        self.assertEqual(first.status_code,200)
        sid = first.json()["session_id"]
        self.assertEqual(self.upload(payload).status_code,409)
        self.assertTrue(self.client.post(f"/stop_video_processing/{sid}").json()["success"])
        second = self.upload(self.video())
        self.assertEqual(second.status_code,200)
        self.assertNotEqual(sid,second.json()["session_id"])
        self.assertEqual(self.finished(second.json()["session_id"])["status"],"completed")

    def test_stream_reconnect_does_not_repeat_inference_and_download_is_repeatable(self):
        sid = self.upload(self.video()).json()["session_id"]
        info = self.finished(sid)
        self.assertEqual(info["status"],"completed",info)
        session = self.manager.items[sid]
        count = session.sequence
        for _ in range(2):
            events = self.client.get(f"/sessions/{sid}/events",headers={"Last-Event-ID":"1"})
            self.assertIn('"terminal": true',events.text)
            image = self.client.get(f"/realtime_video_stream/{sid}")
            self.assertIn(b"Content-Type: image/jpeg",image.content)
            self.assertEqual(self.client.get(f"/download_processed_video/{sid}").status_code,200)
        self.assertEqual(session.sequence,count)
        self.assertFalse(Path(session.source).exists())

    def test_invalid_video_and_missing_gan_are_terminal_and_release_source(self):
        sid = self.upload(b"invalid").json()["session_id"]
        self.assertEqual(self.finished(sid)["status"],"failed")
        with patch("app.models_loader.ensure_gan",return_value=None):
            response = self.client.post("/realtime_video_process",data={"model_type":"underwater","enhancement":"true"},files={"file":("v.mp4",self.video(),"video/mp4")})
            sid = response.json()["session_id"]
            info = self.finished(sid)
            self.assertEqual(info["status"],"failed")
            self.assertIn("Enhancement",info["error"])

    def test_image_never_creates_alarm_state(self):
        _, image = cv2.imencode(".jpg",np.zeros((120,120,3),dtype=np.uint8))
        response = self.client.post("/predict",data={"model_type":"underwater","source":"image"},files={"file":("test.jpg",io.BytesIO(image.tobytes()),"image/jpeg")})
        self.assertEqual(response.status_code,200,response.text)
        self.assertTrue(response.json()["analysis"]["isDrowning"])
        self.assertEqual(self.manager.items,{})
        self.assertNotIn("alarm",response.json()["detections"][0])

    def test_validation_and_unknown_session(self):
        self.assertEqual(self.client.get("/sessions/no-such-session/events").status_code,404)
        self.assertEqual(self.client.post("/webcam/start",data={"confidence_threshold":"nan"}).status_code,400)
        self.assertEqual(self.client.post("/webcam/start",data={"model_type":"invalid"}).status_code,400)

    def test_camera_uses_latest_frame_and_reports_disconnect(self):
        class Camera:
            def __init__(self):
                self.reads, self.released = 0, False
            def isOpened(self):
                return True
            def get(self, prop):
                return 30
            def read(self):
                self.reads += 1
                return (True, np.zeros((120,120,3),dtype=np.uint8)) if self.reads <= 40 else (False,None)
            def release(self):
                self.released = True
        camera = Camera()
        with patch("cv2.VideoCapture", return_value=camera):
            sid = self.client.post("/webcam/start").json()["session_id"]
            info = self.finished(sid)
        self.assertEqual(info["status"],"failed")
        self.assertIn("disconnected",info["error"])
        self.assertTrue(camera.released)
        self.assertLess(self.manager.items[sid].frame_count, camera.reads)
        self.assertFalse(self.client.get("/webcam/status").json()["active"])

    def test_enhancement_video_shares_source_lock_and_downloads_comparison(self):
        with patch("app.models_loader.ensure_gan",return_value=object()), patch("app.detection.enhance_frame",side_effect=lambda frame,gan:(frame.copy(),True)):
            response = self.client.post("/enhance_video", data={"create_comparison":"true"}, files={"file":("v.mp4",self.video(),"video/mp4")})
            self.assertEqual(response.status_code,200,response.text)
            sid = response.json()["session_id"]
            self.assertEqual(self.upload(self.video()).status_code,409)
            self.assertEqual(self.finished(sid)["status"],"completed")
            for kind in ("enhanced","comparison"):
                self.assertEqual(self.client.get(f"/download_enhanced_video/{sid}?video_type={kind}").status_code,200)
            self.assertIn(b"Content-Type: image/jpeg",self.client.get(f"/enhance_video_stream_only/{sid}").content)

if __name__ == "__main__":
    unittest.main()
