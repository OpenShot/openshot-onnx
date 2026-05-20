# Cutie ONNX Exports for OpenShot

This directory builds OpenCV-friendly ONNX exports of
[Cutie](https://github.com/hkchengrex/Cutie) for OpenShot's Object Mask effect.
Cutie is a video object segmentation model by Ho Kei Cheng and contributors.
The upstream Cutie code is MIT licensed. Review the upstream repository and
release notes before redistributing weights or derived ONNX files.

The generated ONNX files and zip packages are intentionally not committed to
Git. Build them locally and upload the zip files as GitHub Release assets.

## Quality Tiers

The OpenShot integration uses fixed 16:9-ish static-shape ONNX exports:

| Tier | Internal size | Stride-16 grid | Notes |
| --- | ---: | ---: | --- |
| Low | `480x272` | `30x17` | Fastest video-shaped tier |
| Medium | `640x368` | `40x23` | Recommended default |
| High | `960x544` | `60x34` | Better edges and thin structures |
| Very High | `1280x720` | `80x45` | Offline/high-quality tier |

All dimensions are divisible by 16. OpenCV DNN is much more reliable with these
static shapes than with dynamic ONNX graphs.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r cutie/requirements.txt
```

You also need OpenCV 4.13.0+ with DNN available through `pkg-config opencv4` if
you want the exporter to compile and run the C++ OpenCV validation probe.

## Build Release Packages

```bash
source .venv/bin/activate
python cutie/scripts/export_cutie_quality_tiers.py
```

The script:

1. Clones the official Cutie repo into `cutie/vendor/Cutie`.
2. Downloads `cutie-base-mega.pth` into `cutie/weights`.
3. Exports the four fixed-size ONNX slices for each quality tier.
4. Simplifies the ONNX graphs.
5. Applies the OpenCV-specific `Unsqueeze` rewrite to the float attention-mask
   memory readout graph.
6. Validates each tier with OpenCV DNN.
7. Writes zip assets under `cutie/releases`.
8. Updates the release catalog at `cutie/models.json`.

Generated local directories:

```text
cutie/vendor/     official Cutie checkout, ignored by Git
cutie/weights/    downloaded upstream checkpoints, ignored by Git
cutie/models/     generated ONNX slices, ignored by Git
cutie/build/      compiled validation probe, ignored by Git
cutie/releases/   generated release zip assets, ignored by Git
```

Build one tier:

```bash
python cutie/scripts/export_cutie_quality_tiers.py --tier medium
```

Reuse an existing Cutie checkout:

```bash
python cutie/scripts/export_cutie_quality_tiers.py \
  --cutie-root /path/to/Cutie \
  --weights /path/to/cutie-base-mega.pth
```

Skip OpenCV validation:

```bash
python cutie/scripts/export_cutie_quality_tiers.py --skip-validate
```

## Release Zip Contents

Each zip contains the four model files needed by libopenshot:

```text
cutie-opencv-medium-640x368.zip
  cutie-encode-key-640x368.onnx
  cutie-encode-value-640x368.onnx
  cutie-memory-readout-floatmask-valid-640x368-m6-topk30-opencv.onnx
  cutie-decode-640x368.onnx
```

The exporter also updates `cutie/models.json` with asset names, dimensions,
checksums, and sizes.

## Individual Scripts

`scripts/export_cutie_slices.py` exports one fixed-size Cutie model set:

```bash
python cutie/scripts/export_cutie_slices.py \
  --cutie-root cutie/vendor/Cutie \
  --weights cutie/weights/cutie-base-mega.pth \
  --output-dir cutie/models \
  --width 640 \
  --height 368
```

`scripts/simplify_cutie_onnx.py` simplifies exported graphs:

```bash
python cutie/scripts/simplify_cutie_onnx.py \
  cutie/models/cutie-encode-key-640x368.onnx \
  cutie/models/cutie-encode-value-640x368.onnx \
  cutie/models/cutie-decode-640x368.onnx
```

For the float attention-mask readout, also write the OpenCV-compatible graph:

```bash
python cutie/scripts/simplify_cutie_onnx.py --opencv-unsqueeze \
  cutie/models/cutie-memory-readout-floatmask-valid-640x368-m6-topk30.onnx
```

## Credit And Licenses

This repository only contains export scripts and integration helpers. It does
not contain the Cutie source code, upstream weights, generated ONNX exports, or
release zip assets.

Important upstream links:

- Cutie GitHub: https://github.com/hkchengrex/Cutie
- Cutie paper/project links: see the official Cutie README
- Cutie release weights: https://github.com/hkchengrex/Cutie/releases
- Cutie license: https://github.com/hkchengrex/Cutie/blob/main/LICENSE
- OpenCV DNN: https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html

The scripts in this directory are MIT licensed as part of this repository.
Upstream Cutie code and model weights remain governed by their own terms.
