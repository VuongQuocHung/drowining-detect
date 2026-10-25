"""Match alarm episodes to independently labeled video events, per person.

Truth JSON: {duration_seconds, video_id, events: [{track_id, start, end}]}.
track_id here is the manually verified system track for the labeled person.
Predictions: session JSONL from the backend. Only red alarm episodes are scored.
"""
import argparse
import json
from pathlib import Path

def score(truth, predictions):
    duration = truth["duration_seconds"]
    if duration <= 0:
        raise ValueError("duration_seconds must be positive")
    episodes, active = [], set()
    for row in predictions:
        timestamp = row.get("source_time")
        if timestamp is None:
            continue
        present = set()
        for person in row["detections"]:
            tid = person["track_id"]
            level = person.get("alert_level")
            if level == "drowning_alert":
                present.add(tid)
                if tid not in active:
                    episodes.append({"track_id": tid, "time": timestamp})
            elif level == "lost" and person.get("previous_level") == "drowning_alert" and tid in active:
                present.add(tid)
        active = present
    matched, delays, false_alarms = set(), [], 0
    for alarm in episodes:
        candidates = [(i,event) for i,event in enumerate(truth["events"]) if
                      str(event["track_id"]) == str(alarm["track_id"]) and event["start"] <= alarm["time"] <= event["end"]]
        if candidates:
            i,event = min(candidates, key=lambda pair:pair[1]["start"])
            if i not in matched:
                delays.append(alarm["time"]-event["start"])
            matched.add(i)
        else:
            false_alarms += 1
    total = len(truth["events"])
    return {"video_id": truth["video_id"], "events": total, "matched": len(matched),
            "missed_event_rate": (total-len(matched))/total if total else None,
            "false_alarms": false_alarms, "false_alarms_per_minute": false_alarms/(duration/60),
            "delay_seconds": delays, "mean_delay_seconds": sum(delays)/len(delays) if delays else None}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    truth = json.loads(Path(args.truth).read_text())
    predictions = [json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line.strip()]
    Path(args.output).write_text(json.dumps(score(truth,predictions),indent=2))
