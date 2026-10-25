"""Observe an actual webcam session for 30 minutes; never fabricate measurements."""
import argparse
import json
import time
import urllib.request
from pathlib import Path

def main(args):
    started = time.monotonic()
    report = {"session_id": args.session, "requested_seconds": args.seconds, "samples": [], "passed": False}
    last_frame = started
    url = f"{args.url}/sessions/{args.session}/events"
    try:
        with urllib.request.urlopen(url,timeout=20) as response:
            for line in response:
                if time.monotonic()-last_frame > 30:
                    raise RuntimeError("No processed frame for 30 seconds")
                if not line.startswith(b"data: "):
                    continue
                event = json.loads(line[6:])
                if event.get("metrics"):
                    last_frame = time.monotonic()
                    report["samples"].append({"source_time":event["source_time"], **event["metrics"]})
                if event.get("terminal"):
                    raise RuntimeError(f"Source ended before soak completed: {event}")
                if time.monotonic()-started >= args.seconds:
                    report["passed"] = bool(report["samples"])
                    break
    except Exception as exc:
        report["error"] = str(exc)
    finally:
        report["elapsed_seconds"] = round(time.monotonic()-started,2)
        samples = report["samples"]
        if samples:
            report["rss_growth_mb"] = samples[-1]["rss_mb"]-samples[0]["rss_mb"]
            report["max_latency_p95_ms"] = max(row["latency_p95_ms"] for row in samples)
        Path(args.output).write_text(json.dumps(report,indent=2))
    if not report["passed"]:
        raise SystemExit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--session", required=True)
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--output", default="soak-report.json")
    main(parser.parse_args())
