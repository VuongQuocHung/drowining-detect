"""Runtime configuration. All paths are independent of the working directory."""
import os
import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "model/detection_model"

def _default_model(view, original):
    # Weights trained by the Kaggle notebook take over when present; delete them to roll back.
    trained = MODELS / f"{view}_selected.pt"
    return str(trained if trained.exists() else MODELS / original)

@dataclass(frozen=True)
class Settings:
    confidence: float = float(os.getenv("CONFIDENCE", "0.5"))
    struggling_seconds: float = float(os.getenv("STRUGGLING_SECONDS", "2"))
    drowning_seconds: float = float(os.getenv("DROWNING_SECONDS", "1"))
    missing_grace: float = float(os.getenv("MISSING_GRACE", "0.5"))
    recovery_seconds: float = float(os.getenv("RECOVERY_SECONDS", "2"))
    alarm_interval: float = float(os.getenv("ALARM_INTERVAL", "10"))
    backend_alarm: bool = os.getenv("BACKEND_ALARM", "0") == "1"
    min_area: float = float(os.getenv("MIN_BOX_AREA", "0"))
    session_ttl: float = float(os.getenv("SESSION_TTL", "3600"))
    underwater_model: str = os.getenv("UNDERWATER_MODEL", _default_model("underwater", "underwaterModel.pt"))
    above_model: str = os.getenv("ABOVE_MODEL", _default_model("abovewater", "abovewaterModel.pt"))

    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError("CONFIDENCE must be between 0 and 1")
        for name in ("struggling_seconds", "drowning_seconds", "missing_grace", "recovery_seconds", "alarm_interval", "session_ttl"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.min_area) or self.min_area < 0:
            raise ValueError("MIN_BOX_AREA must be nonnegative and finite")

settings = Settings()

# Non-person classes are drawn on frames but never tracked or alerted.
OBJECT_CLASSES = ("float",)

def normalize_class(name):
    value = str(name).strip().lower()
    return {"normal": "swimming", "swimmer": "swimming", "swimming": "swimming", "struggling": "struggling",
            "drowning": "drowning", "float": "float"}.get(value, "unknown")
