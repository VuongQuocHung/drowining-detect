"""Compare the same video with GAN off/on and actual models on the local machine."""
import argparse
import json
import platform
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main(args):
    if args.frames <= 0:
        raise ValueError("frames must be positive")
    import cv2
    import torch
    import ultralytics
    from app.models_loader import load_yolo_models
    from app.sessions import SessionManager
    from app.settings import settings
    import shutil

    app = SimpleNamespace(state=SimpleNamespace(models=load_yolo_models(),gan_model=None))
    report = {"source": str(Path(args.video).resolve()), "platform": platform.platform(), "python":platform.python_version(),
              "torch":torch.__version__, "ultralytics":ultralytics.__version__, "opencv":cv2.__version__,
              "device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "config":vars(settings), "runs":[]}
    manager = SessionManager()
    try:
        for gan in (False,True) if args.compare_gan else (False,):
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory)/"input.mp4"
                shutil.copy2(args.video,source)
                session = manager.create(app,args.model_type,str(source),gan,settings.confidence, frame_limit=args.frames)
                session.thread.join()
                records = [e for e in session.events if "metrics" in e]
                report["runs"].append({"enhancement":gan, "status":session.status, "error":session.error,
                    "processed_frames":session.frame_count, "metrics":records[-1]["metrics"] if records else None,
                    "session_id":session.id})
    finally:
        manager.close()
        Path(args.output).write_text(json.dumps(report,indent=2))
    if any(run["status"] == "failed" for run in report["runs"]):
        raise SystemExit(1)

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--video",required=True)
    parser.add_argument("--frames",type=int,default=100)
    parser.add_argument("--model-type",choices=("underwater","above-water"),default="underwater")
    parser.add_argument("--compare-gan",action="store_true")
    parser.add_argument("--output",default="benchmark.json")
    main(parser.parse_args())
