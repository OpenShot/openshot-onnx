#!/usr/bin/env python3
# Copyright (c) 2026 OpenShot Studios, LLC
# SPDX-License-Identifier: MIT
"""Download official YOLO segmentation weights and export OpenCV-friendly ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_URL = "https://github.com/ultralytics/assets/releases/download"
DEFAULT_FAMILIES = ("yolo26", "yolo11", "yolov8")
DEFAULT_SIZES = ("n", "s", "m")
COCO_LABELS = REPO_ROOT / "models" / "labels" / "coco80.names"
YOLOE26_LABELS = REPO_ROOT / "models" / "labels" / "yoloe26-4585.names"
LABEL_MANIFEST = REPO_ROOT / "models" / "labels" / "manifest.json"
MODELS_JSON = REPO_ROOT / "models.json"
DEFAULT_RELEASE_DIR = REPO_ROOT / "releases"
DEFAULT_RELEASE_TAG = "v0.2.0"
DEFAULT_RELEASE_BASE_URL = (
    "https://github.com/OpenShot/openshot-onnx/releases/download"
)

SIZE_NAMES = {
    "n": "Nano",
    "s": "Small",
    "m": "Medium",
    "l": "Large",
    "x": "XLarge",
}

FAMILY_NAMES = {
    "yolo26": "YOLO26",
    "yolo11": "YOLOv11",
    "yolov8": "YOLOv8",
    "yoloe26": "YOLOE-26",
}

MODEL_DESCRIPTIONS = {
    "yolo26n-seg": "recommended, fast",
    "yolo26s-seg": "balanced",
    "yolo26m-seg": "quality",
    "yolo11n-seg": "legacy, fast",
    "yolo11s-seg": "legacy, balanced",
    "yolo11m-seg": "legacy, quality",
    "yolov8n-seg": "legacy, fast",
    "yolov8s-seg": "legacy, balanced",
    "yolov8m-seg": "legacy, quality",
    "yoloe-26n-seg": "more labels, fast",
    "yoloe-26s-seg": "more labels, balanced",
    "yoloe-26m-seg": "more labels, quality",
}

FAMILY_ORDER = ("yolo26", "yoloe26", "yolo11", "yolov8")
SIZE_ORDER = ("n", "s", "m", "l", "x")


@dataclass(frozen=True)
class Family:
    release: str
    url_base: str
    filename_prefix: str
    yoloe: bool = False
    vocab_filename_prefix: str | None = None

    def filename(self, size: str) -> str:
        return f"{self.filename_prefix}{size}-seg.pt"

    def vocab_filename(self, size: str) -> str:
        prefix = self.vocab_filename_prefix or self.filename_prefix
        return f"{prefix}{size}-seg-pf.pt"

    def url(self, size: str) -> str:
        return f"{self.url_base}/{self.release}/{self.filename(size)}"

    def vocab_url(self, size: str) -> str:
        return f"{self.url_base}/{self.release}/{self.vocab_filename(size)}"


FAMILIES = {
    "yolo26": Family("v8.4.0", ASSETS_URL, "yolo26"),
    "yolo11": Family("v8.3.0", ASSETS_URL, "yolo11"),
    "yolov8": Family("v8.4.0", ASSETS_URL, "yolov8"),
    "yoloe26": Family("v8.4.0", ASSETS_URL, "yoloe-26", yoloe=True),
}


@contextmanager
def chdir(path: Path):
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def download(url: str, destination: Path, force: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0 and not force:
        print(f"Using existing {destination}")
        return

    tmp = destination.with_suffix(destination.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(destination)
    print(f"Saved {destination} ({destination.stat().st_size / 1024 / 1024:.1f} MB)")


def validate_onnx(path: Path, validate_opencv: bool) -> None:
    import onnx
    import onnxruntime as ort

    model = onnx.load(path)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_shapes = {i.name: i.shape for i in session.get_inputs()}
    output_shapes = {o.name: o.shape for o in session.get_outputs()}
    print(f"ONNX Runtime loaded {path.name}")
    print(f"  inputs:  {input_shapes}")
    print(f"  outputs: {output_shapes}")

    if validate_opencv:
        import cv2

        net = cv2.dnn.readNetFromONNX(str(path))
        layer_count = len(net.getLayerNames())
        print(f"OpenCV DNN loaded {path.name} ({layer_count} layers)")


def write_label_set(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(names) + "\n"
    if path.exists() and path.read_text() == content:
        print(f"Using existing labels {path}")
        return
    path.write_text(content)
    print(f"Wrote {len(names)} labels to {path}")


def update_label_manifest(model_path: Path, labels_path: Path, names: list[str]) -> None:
    LABEL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    if LABEL_MANIFEST.exists():
        manifest = json.loads(LABEL_MANIFEST.read_text())
    else:
        manifest = {"models": {}}

    manifest["models"][model_path.name] = {
        "labels": labels_path.relative_to(REPO_ROOT).as_posix(),
        "class_count": len(names),
        "output0_width": 4 + len(names) + 32,
    }
    LABEL_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Updated label manifest {LABEL_MANIFEST}")


def record_labels(model_path: Path, labels_path: Path, names: list[str]) -> None:
    write_label_set(labels_path, names)
    update_label_manifest(model_path, labels_path, names)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_labels_path(model_path: Path) -> Path:
    if not LABEL_MANIFEST.exists():
        raise FileNotFoundError(f"Label manifest not found: {LABEL_MANIFEST}")

    manifest = json.loads(LABEL_MANIFEST.read_text())
    model_entry = manifest.get("models", {}).get(model_path.name)
    if not model_entry:
        raise KeyError(f"No label manifest entry for {model_path.name}")

    return REPO_ROOT / model_entry["labels"]


def release_model_defaults(model_id: str) -> dict[str, object]:
    for family_name, family in FAMILIES.items():
        prefix = family.filename_prefix
        for size, size_name in SIZE_NAMES.items():
            if model_id == f"{prefix}{size}-seg":
                family_label = FAMILY_NAMES[family_name]
                entry: dict[str, object] = {
                    "id": model_id,
                    "name": f"{family_label}: {size_name}",
                    "description": MODEL_DESCRIPTIONS.get(model_id, ""),
                    "asset": f"{model_id}.zip",
                }
                if model_id == "yolo26n-seg":
                    entry["recommended"] = True
                return entry
    return {
        "id": model_id,
        "name": model_id,
        "description": "",
        "asset": f"{model_id}.zip",
    }


def model_sort_key(item: dict[str, object]) -> tuple[bool, int, int, str]:
    model_id = str(item["id"])
    for family_index, family_name in enumerate(FAMILY_ORDER):
        prefix = FAMILIES[family_name].filename_prefix
        for size_index, size in enumerate(SIZE_ORDER):
            if model_id == f"{prefix}{size}-seg":
                return (
                    not item.get("recommended", False),
                    family_index,
                    size_index,
                    model_id,
                )
    return (not item.get("recommended", False), len(FAMILY_ORDER), len(SIZE_ORDER), model_id)


def package_model(model_path: Path, release_dir: Path) -> dict[str, object]:
    labels_path = model_labels_path(model_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Label file not found: {labels_path}")

    release_dir.mkdir(parents=True, exist_ok=True)
    model_id = model_path.stem
    zip_path = release_dir / f"{model_id}.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(model_path, "model.onnx")
        archive.write(labels_path, "classes.names")

    entry = release_model_defaults(model_id)
    entry["asset"] = zip_path.name
    entry["sha256"] = sha256_file(zip_path)
    entry["bytes"] = zip_path.stat().st_size
    print(f"Wrote release package {zip_path}")
    return entry


def update_release_manifest(
    packaged_models: list[dict[str, object]],
    models_json: Path,
    release_tag: str,
    base_url: str,
) -> None:
    existing_models: dict[str, dict[str, object]] = {}
    if models_json.exists():
        existing = json.loads(models_json.read_text())
        for model in existing.get("models", []):
            existing_models[model["id"]] = model

    packaged_by_id = {str(model["id"]): model for model in packaged_models}
    all_model_ids = set(existing_models) | set(packaged_by_id)
    updated_models = []
    for model_id in all_model_ids:
        packaged = packaged_by_id.get(model_id)
        defaults = release_model_defaults(model_id)
        previous = existing_models.get(model_id, {})
        entry = {
            "id": model_id,
            "name": previous.get("name", defaults["name"]),
            "description": previous.get("description", defaults["description"]),
            "asset": previous.get("asset", defaults["asset"]),
        }
        if packaged:
            entry["asset"] = packaged["asset"]
            entry["sha256"] = packaged["sha256"]
        elif previous.get("sha256"):
            entry["sha256"] = previous["sha256"]
        if packaged and packaged.get("bytes") is not None:
            entry["bytes"] = packaged["bytes"]
        elif previous.get("bytes") is not None:
            entry["bytes"] = previous["bytes"]
        if previous.get("recommended", defaults.get("recommended", False)):
            entry["recommended"] = True
        updated_models.append(entry)

    models_json.write_text(
        json.dumps(
            {
                "version": release_tag.removeprefix("v"),
                "release": release_tag,
                "base_url": f"{base_url.rstrip('/')}/{release_tag}",
                "models": sorted(updated_models, key=model_sort_key),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Updated release manifest {models_json}")


def model_names(model: Any) -> list[str]:
    names = getattr(model.model, "names", None) or getattr(model, "names", None)
    if isinstance(names, dict):
        return [names[i] for i in range(len(names))]
    return list(names or [])


def export_model(
    weights: Path,
    output_dir: Path,
    imgsz: int,
    opset: int,
    simplify: bool,
    validate_opencv: bool,
) -> Path:
    from ultralytics import YOLO

    print(f"Exporting {weights.name} with end2end=False, nms=False, dynamic=False")
    model = YOLO(str(weights))

    if hasattr(model.model, "end2end"):
        model.model.end2end = False

    exported = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
            dynamic=False,
            nms=False,
            half=False,
            int8=False,
            batch=1,
            device="cpu",
            end2end=False,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / exported.name
    if exported.resolve() != final.resolve():
        shutil.move(str(exported), final)
    record_labels(final, COCO_LABELS, model_names(model))
    validate_onnx(final, validate_opencv)
    return final


def load_yoloe_vocab(vocab_weights: Path) -> list[str]:
    from ultralytics import YOLOE

    vocab_model = YOLOE(str(vocab_weights))
    names = vocab_model.model.names
    if isinstance(names, dict):
        return [names[i] for i in range(len(names))]
    return list(names)


def export_yoloe_model(
    weights: Path,
    vocab_weights: Path,
    text_model_dir: Path,
    output_dir: Path,
    imgsz: int,
    opset: int,
    simplify: bool,
    validate_opencv: bool,
) -> Path:
    from ultralytics import YOLOE

    classes = load_yoloe_vocab(vocab_weights)
    print(
        f"Exporting {weights.name} with {len(classes)} MobileCLIP-embedded classes, "
        "end2end=False, nms=False, dynamic=False"
    )
    model = YOLOE(str(weights))

    if hasattr(model.model, "end2end"):
        model.model.end2end = False

    # Ultralytics downloads MobileCLIP TorchScript weights by relative name.
    # Keep that large export-time dependency out of the repo root.
    with chdir(text_model_dir):
        embeddings = model.model.get_text_pe(classes, cache_clip_model=True)
    model.set_classes(classes, embeddings)

    exported = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
            dynamic=False,
            nms=False,
            half=False,
            int8=False,
            batch=1,
            device="cpu",
            end2end=False,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / exported.name
    if exported.resolve() != final.resolve():
        shutil.move(str(exported), final)
    record_labels(final, YOLOE26_LABELS, classes)
    validate_onnx(final, validate_opencv)
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        nargs="+",
        default=list(DEFAULT_FAMILIES),
        choices=tuple(FAMILIES),
        help="model families to download and export",
    )
    parser.add_argument("--sizes", nargs="+", default=list(DEFAULT_SIZES), choices=("n", "s", "m", "l", "x"))
    parser.add_argument("--imgsz", type=int, default=640, help="static square input size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset for OpenCV DNN compatibility")
    parser.add_argument("--pt-dir", type=Path, default=REPO_ROOT / "models" / "pt")
    parser.add_argument("--onnx-dir", type=Path, default=REPO_ROOT / "models" / "onnx")
    parser.add_argument("--text-model-dir", type=Path, default=REPO_ROOT / "models" / "text")
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--models-json", type=Path, default=MODELS_JSON)
    parser.add_argument("--release-tag", default=DEFAULT_RELEASE_TAG)
    parser.add_argument("--release-base-url", default=DEFAULT_RELEASE_BASE_URL)
    parser.add_argument("--force-download", action="store_true", help="redownload .pt files")
    parser.add_argument(
        "--skip-release-packages",
        action="store_true",
        help="skip release zip creation and models.json updates",
    )
    parser.add_argument("--no-simplify", action="store_true", help="skip ONNX simplification")
    parser.add_argument(
        "--skip-opencv-validate",
        action="store_true",
        help="skip cv2.dnn.readNetFromONNX validation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    produced: list[Path] = []

    for family_name in args.families:
        family = FAMILIES[family_name]
        for size in args.sizes:
            weights = args.pt_dir / family.filename(size)
            download(family.url(size), weights, force=args.force_download)
            if family.yoloe:
                vocab_weights = args.pt_dir / family.vocab_filename(size)
                download(family.vocab_url(size), vocab_weights, force=args.force_download)
                produced.append(
                    export_yoloe_model(
                        weights=weights,
                        vocab_weights=vocab_weights,
                        text_model_dir=args.text_model_dir,
                        output_dir=args.onnx_dir,
                        imgsz=args.imgsz,
                        opset=args.opset,
                        simplify=not args.no_simplify,
                        validate_opencv=not args.skip_opencv_validate,
                    )
                )
            else:
                produced.append(
                    export_model(
                        weights=weights,
                        output_dir=args.onnx_dir,
                        imgsz=args.imgsz,
                        opset=args.opset,
                        simplify=not args.no_simplify,
                        validate_opencv=not args.skip_opencv_validate,
                    )
                )

    print("\nProduced ONNX files:")
    for path in produced:
        print(f"  {path}")

    if not args.skip_release_packages:
        packaged = [package_model(path, args.release_dir) for path in produced]
        update_release_manifest(
            packaged_models=packaged,
            models_json=args.models_json,
            release_tag=args.release_tag,
            base_url=args.release_base_url,
        )
        print("\nProduced release packages:")
        for model in packaged:
            print(f"  {args.release_dir / str(model['asset'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
