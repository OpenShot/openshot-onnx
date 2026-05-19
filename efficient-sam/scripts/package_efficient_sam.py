#!/usr/bin/env python3
# Copyright (c) 2026 OpenShot Studios, LLC
# SPDX-License-Identifier: MIT
"""Download, validate, and package EfficientSAM ONNX assets for OpenShot."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EFFICIENT_SAM_DIR = SCRIPT_DIR.parent
DEFAULT_MODELS_DIR = EFFICIENT_SAM_DIR / "models"
DEFAULT_RELEASE_DIR = EFFICIENT_SAM_DIR / "releases"
DEFAULT_RELEASE_TAG = "v0.1.0"
DEFAULT_RELEASE_BASE_URL = (
    "https://github.com/OpenShot/openshot-onnx/releases/download"
)


@dataclass(frozen=True)
class ModelVariant:
    id: str
    name: str
    description: str
    source_url: str
    source_name: str
    source_sha256: str
    model_name: str
    requires_static_conversion: bool = False
    recommended: bool = False


VARIANTS = {
    "tiny": ModelVariant(
        id="efficient-sam-tiny-1024",
        name="EfficientSAM: Tiny 1024",
        description="Fast prompted seed-mask generator for Object Mask",
        source_url=(
            "https://huggingface.co/opencv/image_segmentation_efficientsam/resolve/main/"
            "image_segmentation_efficientsam_ti_2025april.onnx"
        ),
        source_name="image_segmentation_efficientsam_ti_2025april.onnx",
        source_sha256="4eb496e0a7259d435b49b66faf1754aa45a5c382a34558ddda9a8c6fe5915d77",
        model_name="image_segmentation_efficientsam_ti_2025april.onnx",
        recommended=True,
    ),
    "small": ModelVariant(
        id="efficient-sam-small-static-1024",
        name="EfficientSAM: Small 1024",
        description="Higher-quality prompted seed-mask generator for Object Mask",
        source_url="https://huggingface.co/yunyangx/EfficientSAM/resolve/main/efficientsam_s.onnx",
        source_name="efficientsam_s.onnx",
        source_sha256="b257787eeecdfd0db0626f83a8241874c35c74eb4c25c4d12ff0a478f90f30f9",
        model_name="image_segmentation_efficientsam_s_static_1024.onnx",
        requires_static_conversion=True,
    ),
}


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_sha256: str, force: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0 and not force:
        if sha256_file(destination) == expected_sha256:
            print(f"Using existing {destination}")
            return

    tmp = destination.with_suffix(destination.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(destination)

    actual = sha256_file(destination)
    if actual != expected_sha256:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {destination}: {actual}")
    print(f"Saved {destination} ({destination.stat().st_size / 1024 / 1024:.1f} MB)")


def ensure_opencv_probe(probe: Path) -> None:
    source = SCRIPT_DIR / "efficient_sam_opencv_probe.cpp"
    if probe.exists() and probe.stat().st_mtime >= source.stat().st_mtime:
        return
    probe.parent.mkdir(parents=True, exist_ok=True)
    cflags = subprocess.check_output(["pkg-config", "--cflags", "opencv4"], text=True).split()
    libs = subprocess.check_output(["pkg-config", "--libs", "opencv4"], text=True).split()
    run(["g++", "-std=c++17", str(source), "-o", str(probe), *cflags, *libs])


def validate_model(probe: Path, model_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = Path("experiments/xmem_onnx_temp/XMem_Export/sample/test-sample1-1frame.png")
    if sample.exists():
        run([str(probe), str(model_path), str(sample), str(output_dir / "dog_point"), "570,1180,1"])
        return

    run([str(probe), str(model_path), str(model_path), str(output_dir / "noop"), "1,1,1"])


def convert_static_1024(source_path: Path, model_path: Path, force: bool = False) -> None:
    if model_path.exists() and model_path.stat().st_size > 0 and not force:
        print(f"Using existing {model_path}")
        return
    run([
        sys.executable,
        "-m",
        "onnxsim",
        str(source_path),
        str(model_path),
        "--overwrite-input-shape",
        "batched_images:1,3,1024,1024",
        "batched_point_coords:1,1,6,2",
        "batched_point_labels:1,1,6",
    ])


def package_model(variant: ModelVariant, model_path: Path, release_dir: Path) -> dict[str, object]:
    release_dir.mkdir(parents=True, exist_ok=True)
    zip_path = release_dir / f"{variant.id}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(model_path, variant.model_name)

    entry = {
        "id": variant.id,
        "name": variant.name,
        "description": variant.description,
        "asset": zip_path.name,
        "sha256": sha256_file(zip_path),
        "bytes": zip_path.stat().st_size,
        "model": variant.model_name,
        "source_model": variant.source_name,
        "model_sha256": sha256_file(model_path),
        "input_size": [1024, 1024],
        "recommended": variant.recommended,
    }
    print(f"Wrote {zip_path}")
    return entry


def write_manifest(release_dir: Path, release_tag: str, release_base_url: str, entries: list[dict[str, object]]) -> None:
    manifest = {
        "schema": 1,
        "name": "EfficientSAM OpenCV ONNX Models",
        "release_tag": release_tag,
        "download_base_url": f"{release_base_url}/{release_tag}",
        "models": entries,
    }
    path = release_dir / "efficient-sam-models.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--release-tag", default=DEFAULT_RELEASE_TAG)
    parser.add_argument("--release-base-url", default=DEFAULT_RELEASE_BASE_URL)
    parser.add_argument(
        "--variant",
        choices=[*VARIANTS.keys(), "all"],
        default="all",
        help="EfficientSAM model variant to package.",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    args.models_dir = args.models_dir.expanduser().resolve()
    args.release_dir = args.release_dir.expanduser().resolve()

    selected = VARIANTS.values() if args.variant == "all" else [VARIANTS[args.variant]]
    entries = []
    probe = EFFICIENT_SAM_DIR / "build" / "efficient_sam_opencv_probe"
    if not args.skip_validate:
        ensure_opencv_probe(probe)

    for variant in selected:
        source_path = args.models_dir / variant.source_name
        model_path = args.models_dir / variant.model_name

        if not args.skip_download:
            download(variant.source_url, source_path, variant.source_sha256, args.force_download)
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        if variant.requires_static_conversion:
            convert_static_1024(source_path, model_path, args.force_convert)
        elif source_path != model_path:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, model_path)

        if not model_path.exists():
            raise FileNotFoundError(model_path)

        if not args.skip_validate:
            validate_model(probe, model_path, args.release_dir / "_validate" / variant.id)
        entries.append(package_model(variant, model_path, args.release_dir))

    write_manifest(args.release_dir, args.release_tag, args.release_base_url, entries)


if __name__ == "__main__":
    main()
