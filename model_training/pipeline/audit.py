"""Validate YOLO labels and prepare immutable, reproducible split manifests."""
import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm
import yaml

# Internal labels understood by the backend; "normal" is the legacy name for swimming.
CANONICAL = ("swimming", "struggling", "drowning")
ALIASES = {"normal": "swimming"}
COLORS = {"swimming": "lime", "struggling": "yellow", "drowning": "red"}
SPLITS = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}
SOURCE = "https://figshare.com/articles/dataset/Underwater_Drowning_Detection_Dataset/29497235?file=56039351"

def split_dirs(root):
    """(split, folder name, images dir, labels dir) for images/{split}/ (Figshare) or {split}/images/ (Roboflow)."""
    if (root / "images").is_dir():
        found = [(SPLITS[d.name], d.name, d, root / "labels" / d.name) for d in sorted((root / "images").iterdir())
                 if d.is_dir() and d.name in SPLITS]
    else:
        found = [(SPLITS[d.name], d.name, d / "images", d / "labels") for d in sorted(root.iterdir())
                 if d.is_dir() and d.name in SPLITS and (d / "images").is_dir()]
    if not found:
        raise ValueError("Expected images/{train,val,test}/... or {train,valid,test}/images/...")
    return found

def class_names(root, names=None, rename=None):
    """Dataset class order, mapped to internal labels; unknown names must be renamed explicitly."""
    if names is None:
        for candidate in (root / "classes.txt", root / "labels/train/classes.txt", root / "train/labels/classes.txt"):
            if candidate.exists():
                names = [v.strip() for v in candidate.read_text(encoding="utf-8").splitlines() if v.strip()]
                break
        else:
            if not (root / "data.yaml").exists():
                raise ValueError("No classes.txt or data.yaml found; pass the class order with --names")
            names = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"]
            if isinstance(names, dict):
                names = [names[key] for key in sorted(names)]
    rename = rename or {}
    mapped = []
    for name in names:
        value = str(rename.get(name, name)).strip().lower()
        value = ALIASES.get(value, value)
        if value not in CANONICAL:
            raise ValueError(f"Class '{name}' is not swimming/struggling/drowning; map it with --rename '{name}=<label>'")
        mapped.append(value)
    if len(set(mapped)) != len(mapped) or "drowning" not in mapped:
        raise ValueError(f"Classes {names} must map to distinct labels including drowning, got {mapped}")
    return mapped

def audit(root, output, groups=None, names=None, rename=None, source=SOURCE, drop_exact_duplicates=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    groups_map = {}
    if groups:
        with open(groups, newline="", encoding="utf-8-sig") as stream:
            groups_map = {row["image"]: row["group"] for row in csv.DictReader(stream)}
        if len(set(groups_map.values())) < 3:
            raise ValueError("At least three independent source groups are required")
    names = class_names(root, names, rename)
    records, errors, samples = [], [], []
    print("Scanning image paths...", flush=True)
    paths = [(split, folder, images, labels, path) for split, folder, images, labels in split_dirs(root)
             for path in sorted(images.rglob("*")) if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png")]
    for split, folder, images, labels, path in tqdm(paths, desc="[1/2] Images + labels", unit="image", mininterval=1, dynamic_ncols=True):
        # Group CSV keys stay "<split folder>/<file>", e.g. train/frame_1.jpg or valid/frame_1.jpg.
        rel = Path(folder) / path.relative_to(images)
        label = (labels / path.relative_to(images)).with_suffix(".txt")
        try:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                width, height = image.size
                pixel_hash = hashlib.sha256(image.tobytes() + f"{width},{height}".encode()).hexdigest()
                tiny = image.convert("L").resize((9, 8))
                pixels = tiny.tobytes()
                dhash = sum((pixels[y*9+x] > pixels[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))
                boxes = []
                for line in label.read_text().splitlines():
                    if not line.strip():
                        continue
                    values = list(map(float, line.split()))
                    if len(values) != 5 or not all(math.isfinite(v) for v in values):
                        raise ValueError("Expected five finite YOLO fields")
                    cid, x, y, w, h = values
                    if cid not in range(len(names)) or not (0 < w <= 1 and 0 < h <= 1 and 0 <= x <= 1 and 0 <= y <= 1):
                        raise ValueError("Invalid class or normalized coordinates")
                    if x-w/2 < -1e-5 or y-h/2 < -1e-5 or x+w/2 > 1.00001 or y+h/2 > 1.00001:
                        raise ValueError("Bounding box extends beyond image")
                    boxes.append(values)
                group = groups_map.get(rel.as_posix()) if groups else None
                if groups and not group:
                    raise ValueError("Missing group mapping")
                records.append({"image": str(path), "relative_image": rel.as_posix(), "label": str(label),
                                "split": split, "group": group, "sha256": pixel_hash, "dhash": dhash,
                                "width": width, "height": height, "classes": [int(b[0]) for b in boxes]})
                if len(samples) < 12:
                    draw = ImageDraw.Draw(image)
                    for cid,x,y,w,h in boxes:
                        draw.rectangle(((x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height), outline=COLORS[names[int(cid)]], width=3)
                        draw.text(((x-w/2)*width,max(0,(y-h/2)*height)), names[int(cid)], fill="white")
                    samples.append(ImageOps.pad(image, (320,320)))
        except Exception as exc:
            errors.append({"image": str(path), "error": str(exc)})
    if not records:
        errors.append({"error": "No valid images found"})
    if groups and records:
        unique = sorted({r["group"] for r in records})
        random.Random(42).shuffle(unique)
        train_end = min(len(unique)-2, max(1, round(len(unique)*.7)))
        val_end = min(len(unique)-1, max(train_end+1, round(len(unique)*.85)))
        split_groups = {group: "train" if i < train_end else "val" if i < val_end else "test" for i,group in enumerate(unique)}
        for record in records:
            record["split"] = split_groups[record["group"]]
    exact, near = [], []
    # 5,613 images: pairwise dHash check is bounded and requires no extra index dependency.
    total_pairs = len(records) * (len(records)-1) // 2
    with tqdm(total=total_pairs, desc="[2/2] Duplicate scan", unit="pair", mininterval=1, dynamic_ncols=True) as progress:
        for i, left in enumerate(records):
            for right in records[i+1:]:
                if left["split"] == right["split"]:
                    continue
                pair = [left["relative_image"], right["relative_image"]]
                if left["sha256"] == right["sha256"]:
                    exact.append(pair)
                elif (left["dhash"] ^ right["dhash"]).bit_count() <= 4:
                    near.append(pair)
            # Batch updates keep the progress bar cheap even for millions of pairs.
            progress.set_postfix(exact=len(exact), near=len(near), refresh=False)
            progress.update(len(records)-i-1)
    dropped = []
    if drop_exact_duplicates and exact:
        # Keep the copy in the earliest split (train > val > test); listed in the report, source files untouched.
        rank = {"train": 0, "val": 1, "test": 2}
        split_of = {r["relative_image"]: r["split"] for r in records}
        dropped = sorted({max(pair, key=lambda image: rank[split_of[image]]) for pair in exact})
        records = [r for r in records if r["relative_image"] not in set(dropped)]
        exact = [pair for pair in exact if not set(pair) & set(dropped)]
    print("Saving audit.json, manifest.json and annotation preview...", flush=True)
    report = {"source": source, "names": names, "total_images": len(records), "errors": errors,
              "split_counts": dict(Counter(r["split"] for r in records)),
              "class_boxes": dict(Counter(names[c] for r in records for c in r["classes"])),
              "split_class_boxes": {split: dict(Counter(names[c] for r in records if r["split"] == split for c in r["classes"])) for split in ("train","val","test")},
              "exact_cross_split": exact, "near_cross_split": near, "dropped_exact_duplicates": dropped,
              "grouped": bool(groups), "limitations": [] if groups else ["Source groups unknown; original split retained. Independent labeled video evaluation required."]}
    (output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    if samples:
        canvas = Image.new("RGB",(1280,math.ceil(len(samples)/4)*320))
        for i, sample in enumerate(samples):
            canvas.paste(sample,((i%4)*320,(i//4)*320))
        canvas.save(output / "annotations.jpg")
    if errors or exact:
        raise ValueError("Dataset validation failed: inspect audit.json; fix invalid labels/exact cross-split duplicates")
    data = {"names": dict(enumerate(names)), "path": str(root)}
    for split in ("train","val","test"):
        paths = [r["image"] for r in records if r["split"] == split]
        if paths:
            listing = output / f"{split}.txt"
            listing.write_text("\n".join(paths)+"\n", encoding="utf-8")
            data[split] = str(listing)
    if not all(split in data for split in ("train","val")):
        raise ValueError("Both train and val splits are required")
    (output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", help="CSV with image (relative to images/) and group (source session) columns")
    parser.add_argument("--names", help="Comma-separated class order when the dataset has no classes.txt/data.yaml")
    parser.add_argument("--rename", action="append", default=[], metavar="OLD=NEW",
                        help="Map a dataset class name to swimming/struggling/drowning; repeatable")
    parser.add_argument("--source", default=SOURCE, help="Dataset URL recorded in audit.json")
    parser.add_argument("--drop-exact-duplicates", action="store_true",
                        help="Exclude val/test copies of images identical to another split instead of failing")
    args = parser.parse_args()
    rename = dict(item.split("=", 1) for item in args.rename)
    report = audit(args.root, args.output, args.groups, args.names and args.names.split(","), rename, args.source,
                   args.drop_exact_duplicates)
    # Large duplicate lists belong in the report, not the notebook output stream.
    summary = {key: value for key, value in report.items()
               if key not in ("errors", "exact_cross_split", "near_cross_split", "dropped_exact_duplicates")}
    summary.update(errors=len(report["errors"]), exact_cross_split=len(report["exact_cross_split"]),
                   near_cross_split=len(report["near_cross_split"]),
                   dropped_exact_duplicates=len(report["dropped_exact_duplicates"]),
                   report=str(Path(args.output).resolve() / "audit.json"))
    print(json.dumps(summary, indent=2), flush=True)
