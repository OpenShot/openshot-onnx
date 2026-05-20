#!/usr/bin/env python3
# Copyright (c) 2026 OpenShot Studios, LLC
# SPDX-License-Identifier: MIT
"""Build OpenCV-friendly Cutie ONNX release packages for OpenShot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CUTIE_DIR = SCRIPT_DIR.parent
DEFAULT_CUTIE_ROOT = CUTIE_DIR / "vendor" / "Cutie"
DEFAULT_WEIGHTS = CUTIE_DIR / "weights" / "cutie-base-mega.pth"
DEFAULT_MODELS_DIR = CUTIE_DIR / "models"
DEFAULT_RELEASE_DIR = CUTIE_DIR / "releases"
DEFAULT_MODELS_JSON = CUTIE_DIR / "models.json"
DEFAULT_RELEASE_TAG = "v0.2.0"
DEFAULT_RELEASE_BASE_URL = (
    "https://github.com/OpenShot/openshot-onnx/releases/download"
)

CUTIE_REPO_URL = "https://github.com/hkchengrex/Cutie.git"
CUTIE_WEIGHTS_URL = (
    "https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth"
)
CUTIE_WEIGHTS_MD5 = "a6071de6136982e396851903ab4c083a"


@dataclass(frozen=True)
class Tier:
    id: str
    name: str
    width: int
    height: int
    description: str

    @property
    def size_name(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def package_name(self) -> str:
        return f"cutie-opencv-{self.id}-{self.size_name}.zip"


TIERS = (
    Tier("low", "Low", 480, 272, "Fastest 16:9 Cutie propagation tier"),
    Tier("medium", "Medium", 640, 368, "Balanced 16:9 Cutie propagation tier"),
    Tier("high", "High", 960, 544, "Higher-detail 16:9 Cutie propagation tier"),
    Tier("very-high", "Very High", 1280, 720, "Highest-detail 16:9 Cutie propagation tier"),
)


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, md5: str | None = None, force: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0 and not force:
        if md5 is None or file_md5(destination) == md5:
            print(f"Using existing {destination}")
            return

    tmp = destination.with_suffix(destination.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(destination)

    if md5 and file_md5(destination) != md5:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {destination}")
    print(f"Saved {destination} ({destination.stat().st_size / 1024 / 1024:.1f} MB)")


def ensure_cutie_checkout(path: Path) -> None:
    if (path / "cutie" / "model" / "cutie.py").exists():
        print(f"Using existing Cutie checkout {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", CUTIE_REPO_URL, str(path)])


def ensure_opencv_probe(probe: Path) -> None:
    source = SCRIPT_DIR / "cutie_opencv_probe.cpp"
    if probe.exists() and probe.stat().st_mtime >= source.stat().st_mtime:
        return
    probe.parent.mkdir(parents=True, exist_ok=True)
    cflags = subprocess.check_output(["pkg-config", "--cflags", "opencv4"], text=True).split()
    libs = subprocess.check_output(["pkg-config", "--libs", "opencv4"], text=True).split()
    run(["g++", "-std=c++17", str(source), "-o", str(probe), *cflags, *libs])


def export_tier(args: argparse.Namespace, tier: Tier) -> None:
    run([
        sys.executable,
        str(SCRIPT_DIR / "export_cutie_slices.py"),
        "--cutie-root",
        str(args.cutie_root),
        "--weights",
        str(args.weights),
        "--output-dir",
        str(args.models_dir),
        "--width",
        str(tier.width),
        "--height",
        str(tier.height),
        "--memory-frames",
        str(args.memory_frames),
        "--top-k",
        str(args.top_k),
        "--opset",
        str(args.opset),
    ])


def simplify_tier(args: argparse.Namespace, tier: Tier) -> None:
    size = tier.size_name
    common = [
        args.models_dir / f"cutie-encode-key-{size}.onnx",
        args.models_dir / f"cutie-encode-value-{size}.onnx",
        args.models_dir / f"cutie-decode-{size}.onnx",
    ]
    run([sys.executable, str(SCRIPT_DIR / "simplify_cutie_onnx.py"), *map(str, common)])
    run([
        sys.executable,
        str(SCRIPT_DIR / "simplify_cutie_onnx.py"),
        "--opencv-unsqueeze",
        str(args.models_dir / f"cutie-memory-readout-floatmask-valid-{size}-m{args.memory_frames}-topk{args.top_k}.onnx"),
    ])


def tier_files(models_dir: Path, tier: Tier, memory_frames: int, top_k: int) -> list[Path]:
    size = tier.size_name
    return [
        models_dir / f"cutie-encode-key-{size}-sim.onnx",
        models_dir / f"cutie-encode-value-{size}-sim.onnx",
        models_dir / f"cutie-memory-readout-floatmask-valid-{size}-m{memory_frames}-topk{top_k}-opencv.onnx",
        models_dir / f"cutie-decode-{size}-sim.onnx",
    ]


def package_tier(args: argparse.Namespace, tier: Tier) -> dict[str, object]:
    args.release_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.release_dir / tier.package_name
    archive_names = [
        f"cutie-encode-key-{tier.size_name}.onnx",
        f"cutie-encode-value-{tier.size_name}.onnx",
        f"cutie-memory-readout-floatmask-valid-{tier.size_name}-m{args.memory_frames}-topk{args.top_k}-opencv.onnx",
        f"cutie-decode-{tier.size_name}.onnx",
    ]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, archive_name in zip(tier_files(args.models_dir, tier, args.memory_frames, args.top_k), archive_names):
            if not source.exists():
                raise FileNotFoundError(source)
            archive.write(source, archive_name)

    print(f"Wrote {zip_path}")
    return {
        "id": f"cutie-{tier.id}",
        "name": f"Cutie: {tier.name}",
        "description": tier.description,
        "asset": zip_path.name,
        "sha256": file_sha256(zip_path),
        "bytes": zip_path.stat().st_size,
        **({"recommended": True} if tier.id == "medium" else {}),
    }


def validate_tier(args: argparse.Namespace, tier: Tier, probe: Path) -> None:
    staging = args.release_dir / f"_validate_{tier.id}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    archive_names = [
        f"cutie-encode-key-{tier.size_name}.onnx",
        f"cutie-encode-value-{tier.size_name}.onnx",
        f"cutie-memory-readout-floatmask-valid-{tier.size_name}-m{args.memory_frames}-topk{args.top_k}-opencv.onnx",
        f"cutie-decode-{tier.size_name}.onnx",
    ]
    for source, name in zip(tier_files(args.models_dir, tier, args.memory_frames, args.top_k), archive_names):
        shutil.copy2(source, staging / name)

    run([str(probe), str(staging), str(tier.width), str(tier.height), str(args.memory_frames)])
    shutil.rmtree(staging)


def model_sort_key(item: dict[str, object]) -> tuple[int, str]:
    order = {f"cutie-{tier.id}": index for index, tier in enumerate(TIERS)}
    model_id = str(item["id"])
    return (order.get(model_id, len(order)), model_id)


def catalog_entry(entry: dict[str, object]) -> dict[str, object]:
    keys = ("id", "name", "description", "asset", "sha256", "bytes", "recommended")
    catalog = {key: entry[key] for key in keys if key in entry}
    if not catalog.get("recommended", False):
        catalog.pop("recommended", None)
    return catalog


def write_manifest(args: argparse.Namespace, entries: list[dict[str, object]]) -> None:
    existing_models: dict[str, dict[str, object]] = {}
    if args.models_json.exists():
        existing = json.loads(args.models_json.read_text(encoding="utf-8"))
        for model in existing.get("models", []):
            existing_models[str(model["id"])] = catalog_entry(model)

    updated_models = {**existing_models}
    for entry in entries:
        updated_models[str(entry["id"])] = catalog_entry(entry)

    manifest = {
        "version": args.release_tag.removeprefix("v"),
        "release": args.release_tag,
        "base_url": f"{args.release_base_url.rstrip('/')}/{args.release_tag}",
        "models": sorted(updated_models.values(), key=model_sort_key),
    }
    args.models_json.parent.mkdir(parents=True, exist_ok=True)
    args.models_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.models_json}")


def selected_tiers(names: list[str]) -> list[Tier]:
    if not names:
        return list(TIERS)
    by_id = {tier.id: tier for tier in TIERS}
    return [by_id[name] for name in names]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutie-root", type=Path, default=DEFAULT_CUTIE_ROOT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--models-json", type=Path, default=DEFAULT_MODELS_JSON)
    parser.add_argument("--release-tag", default=DEFAULT_RELEASE_TAG)
    parser.add_argument("--release-base-url", default=DEFAULT_RELEASE_BASE_URL)
    parser.add_argument("--tier", choices=[tier.id for tier in TIERS], action="append")
    parser.add_argument("--memory-frames", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-clone", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-simplify", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()
    args.cutie_root = args.cutie_root.expanduser().resolve()
    args.weights = args.weights.expanduser().resolve()
    args.models_dir = args.models_dir.expanduser().resolve()
    args.release_dir = args.release_dir.expanduser().resolve()
    args.models_json = args.models_json.expanduser().resolve()

    if not args.skip_export and not args.skip_clone:
        ensure_cutie_checkout(args.cutie_root)
    if not args.skip_export:
        download(CUTIE_WEIGHTS_URL, args.weights, CUTIE_WEIGHTS_MD5, args.force_download)

    probe = CUTIE_DIR / "build" / "cutie_opencv_probe"
    if not args.skip_validate:
        ensure_opencv_probe(probe)

    entries = []
    for tier in selected_tiers(args.tier or []):
        if not args.skip_export:
            export_tier(args, tier)
        if not args.skip_simplify:
            simplify_tier(args, tier)
        if not args.skip_validate:
            validate_tier(args, tier, probe)
        entries.append(package_tier(args, tier))

    write_manifest(args, entries)


if __name__ == "__main__":
    main()
