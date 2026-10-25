"""Per-person temporal decisions, independent of models and wall-clock sleeps."""
from dataclasses import dataclass
from .settings import settings, normalize_class

@dataclass
class Person:
    label: str = "unknown"
    elapsed: float = 0.0
    last_seen: float = 0.0
    last_tick: float = 0.0
    level: str = "normal"
    missing: bool = False
    last_alarm: float = float("-inf")

class AlertEngine:
    def __init__(self, config=settings):
        self.config = config
        self.people = {}

    def update(self, detections, timestamp):
        seen, result = set(), []
        for detection in detections:
            tid = detection["track_id"]
            seen.add(tid)
            label = normalize_class(detection["class"])
            person = self.people.get(tid)
            if person is None:
                person = Person(last_seen=timestamp, last_tick=timestamp)
                self.people[tid] = person
            gap = max(0.0, timestamp - person.last_seen)
            # Missing intervals never contribute evidence. Long gaps reset pending evidence.
            continuous = not person.missing
            if person.label != label or (person.missing and gap > self.config.missing_grace):
                person.elapsed = 0.0
            elif continuous:
                person.elapsed += max(0.0, timestamp - person.last_tick)
            person.label = label
            person.last_seen = person.last_tick = timestamp
            person.missing = False
            previous = person.level
            if label == "drowning" and person.elapsed + 1e-9 >= self.config.drowning_seconds:
                person.level = "drowning_alert"
            elif label == "struggling" and person.elapsed + 1e-9 >= self.config.struggling_seconds:
                if previous != "drowning_alert":
                    person.level = "struggling_alert"
            elif label == "swimming" and person.elapsed + 1e-9 >= self.config.recovery_seconds:
                person.level = "normal"
            elif previous not in ("drowning_alert", "struggling_alert"):
                person.level = "normal" if label == "swimming" else "suspected"
            alarm = person.level == "drowning_alert" and (
                previous != "drowning_alert" or timestamp - person.last_alarm >= self.config.alarm_interval)
            if alarm:
                person.last_alarm = timestamp
            result.append({**detection, "alert_level": person.level, "alarm": alarm, "observed_seconds": round(person.elapsed, 3)})
        for tid, person in list(self.people.items()):
            if tid in seen:
                continue
            person.missing = True
            person.last_tick = timestamp
            if person.label == "swimming" or timestamp - person.last_seen > self.config.missing_grace:
                person.elapsed = 0.0
            result.append({"track_id": tid, "class": "unknown", "confidence": None,
                           "alert_level": "lost", "previous_level": person.level, "alarm": False})
            # Retain lost alerts briefly, but bound memory for long camera runs.
            if timestamp - person.last_seen > 30:
                del self.people[tid]
        return result
