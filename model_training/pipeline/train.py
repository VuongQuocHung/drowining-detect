"""Two controlled experiments, validation-only selection, explicit held-out test."""
import argparse
import hashlib
import json
import platform
import importlib.metadata
import subprocess
import sys
import shutil
import time
from pathlib import Path

def evaluate(model, data, split, output):
    import torch
    import numpy as np
    # Measure repeated inference on the same validation/test image, without plotting.
    import yaml
    config = yaml.safe_load(Path(data).read_text())
    image = Path(config[split]).read_text().splitlines()[0]
    for _ in range(3):
        model.predict(image, imgsz=640, device=0, verbose=False)
    times = []
    for _ in range(20):
        torch.cuda.synchronize()
        started = time.perf_counter()
        model.predict(image, imgsz=640, device=0, verbose=False)
        torch.cuda.synchronize()
        times.append((time.perf_counter()-started)*1000)
    metrics = model.val(data=str(data), split=split, imgsz=640, device=0, batch=4,
                        project=str(output), name=f"{split}_metrics", exist_ok=True, plots=True)
    per_class = {}
    for index, cid in enumerate(metrics.box.ap_class_index):
        per_class[model.names[int(cid)].lower()] = {
            "precision": float(metrics.box.p[index]), "recall": float(metrics.box.r[index]),
            "map50": float(metrics.box.ap50[index]), "map50_95": float(metrics.box.ap[index])}
    if "drowning" not in per_class:
        raise ValueError(f"The {split} split must contain drowning annotations")
    return {"per_class": per_class, "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map), "latency_p50_ms": float(np.median(times)),
            "latency_p95_ms": float(np.percentile(times,95)), "speed": metrics.speed}

def choose(results):
    best_recall = max(r["per_class"]["drowning"]["recall"] for r in results)
    candidates = [r for r in results if best_recall-r["per_class"]["drowning"]["recall"] <= .01]
    best_precision = max(r["per_class"]["drowning"]["precision"] for r in candidates)
    candidates = [r for r in candidates if best_precision-r["per_class"]["drowning"]["precision"] <= .01]
    return min(candidates, key=lambda r:r["latency_p50_ms"])

def main(args):
    import torch
    import ultralytics
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator before training")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = Path(args.data).resolve()
    audit = json.loads((data.parent / "audit.json").read_text())
    if audit["errors"] or audit["exact_cross_split"]:
        raise ValueError("Audit failed")
    for split in ("train", "val"):
        if any(audit["split_class_boxes"].get(split, {}).get(name,0) == 0 for name in audit["names"]):
            raise ValueError(f"Every class in {audit['names']} must be represented in {split}")
    manifest = data.parent / "manifest.json"
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    hardware = {"gpu": torch.cuda.get_device_name(0), "python": platform.python_version(),
                "torch": torch.__version__, "ultralytics": ultralytics.__version__, "manifest_sha256": manifest_hash}
    hardware.update({name: importlib.metadata.version(name) for name in ("torchvision", "numpy", "opencv-python", "Pillow")})
    hardware["cuda"] = torch.version.cuda
    previous_environment = output / "environment.json"
    if previous_environment.exists() and (args.resume or args.test_checkpoint):
        previous = json.loads(previous_environment.read_text())
        if previous["manifest_sha256"] != manifest_hash:
            raise ValueError("Dataset manifest changed; start a new experiment directory")
        for name in ("torch", "torchvision", "numpy", "ultralytics"):
            if name in previous and previous[name] != hardware[name]:
                raise ValueError(f"{name} changed; restore the previous environment or start a new experiment directory")
    for name in ("audit.json", "manifest.json", "data.yaml"):
        shutil.copy2(data.parent / name, output / name)
    (output / "environment.json").write_text(json.dumps(hardware,indent=2))
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (output / "installed-packages.txt").write_text(frozen, encoding="utf-8")
    if args.test_checkpoint:
        import yaml
        if "test" not in yaml.safe_load(data.read_text()):
            raise ValueError("No independent test split available; do not relabel validation as test")
        selected = json.loads((output / "selected.json").read_text())
        checkpoint = Path(args.test_checkpoint).resolve()
        if checkpoint != Path(selected["checkpoint"]).resolve():
            raise ValueError("Test only the checkpoint selected by validation")
        result = evaluate(YOLO(str(checkpoint)), data, "test", output / "held_out")
        (output / "test.json").write_text(json.dumps(result,indent=2))
        return
    results = []
    for name in ("yolo11n", "yolo11s"):
        destination = output / name
        last = destination / "weights/last.pt"
        result_path = destination / "validation.json"
        if args.resume and result_path.exists():
            results.append(json.loads(result_path.read_text()))
            continue
        for batch in (16,8,4):
            try:
                resume = args.resume and last.exists()
                model = YOLO(str(last) if resume else f"{name}.pt")
                if resume:
                    model.train(resume=True, device=0)
                else:
                    model.train(data=str(data), imgsz=640, epochs=100, patience=15, seed=42,
                                batch=batch, device=0, workers=2, project=str(output), name=name,
                                exist_ok=True, save=True, save_period=5, deterministic=True)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch == 4:
                    raise
                # Resume checkpoints retain their original batch size; restart this experiment.
                args.resume = False
        checkpoint = destination / "weights/best.pt"
        result = {"model": name, "checkpoint": str(checkpoint),
                  **evaluate(YOLO(str(checkpoint)), data, "val", destination)}
        result_path.write_text(json.dumps(result,indent=2))
        results.append(result)
        (output / "comparison.json").write_text(json.dumps(results,indent=2))
    selected = choose(results)
    (output / "selected.json").write_text(json.dumps(selected,indent=2))
    shutil.copy2(selected["checkpoint"], output / f"{args.name}_selected.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test-checkpoint")
    parser.add_argument("--name", default="underwater", help="Camera view; the chosen weights become <name>_selected.pt")
    main(parser.parse_args())
