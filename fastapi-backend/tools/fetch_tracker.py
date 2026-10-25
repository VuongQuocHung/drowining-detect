"""Fetch the checkpoint linked by the upstream DeepSORT README (never YOLO weights)."""
import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://drive.usercontent.google.com/download?id=1_qwTWdzT9dWNudpusgKavj_4elGgbkUN&export=download&confirm=t"
SHA256 = "df75ddef42c3d1bda67bc94b093e7ce61de7f75a89f36a8f868a428462198316"

if __name__ == "__main__":
    target = ROOT / "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
    if target.exists():
        print("Already present:", target)
        raise SystemExit(0)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".download")
    try:
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open("wb") as stream:
            while block := response.read(1024*1024):
                stream.write(block)
        if temporary.stat().st_size < 10_000_000:
            raise ValueError("Download is not the expected checkpoint; see upstream README")
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
            raise ValueError("Checkpoint checksum mismatch")
        temporary.replace(target)
        print("SHA256:", hashlib.sha256(target.read_bytes()).hexdigest())
    finally:
        temporary.unlink(missing_ok=True)
