# YOLO ONNX Exports for OpenShot

This repo builds OpenCV-friendly ONNX exports of official Ultralytics YOLO
segmentation models for use in OpenShot and libopenshot.

The generated models are static ONNX graphs: no Python, PyTorch, or Ultralytics
runtime is needed after export. OpenShot/libopenshot can load them with OpenCV
DNN and handle thresholds, NMS, labels, boxes, and masks in C++.

## Getting Started

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps -r requirements.txt

python scripts/export_yolo_seg_onnx.py
```

Default output goes here:

```text
models/pt/       downloaded official checkpoints
models/onnx/     exported ONNX models
models/labels/   shared label files and manifest
releases/        generated GitHub Release zip assets
models.json      friendly release/download catalog for consumers
```

Large model binaries and generated release zips are ignored by Git. `models.json`
is the file apps and scripts should read: it lists the friendly model names,
release zip assets, and download base URL. During release packaging, the exporter
refreshes each generated zip checksum.

## Models

Default export:

```bash
python scripts/export_yolo_seg_onnx.py
```

Exports nano, small, and medium segmentation models for:

- `yolov8` COCO 80 classes
- `yolo11` COCO 80 classes
- `yolo26` COCO 80 classes

Optional broad-vocabulary YOLOE export:

```bash
python scripts/export_yolo_seg_onnx.py --families yoloe26
```

YOLOE-26 uses the official prompt-free checkpoint vocabulary, embeds its 4,585
labels with MobileCLIP during export, and writes a static ONNX model. MobileCLIP
is not needed at runtime.

Export everything:

```bash
python scripts/export_yolo_seg_onnx.py --families yolov8 yolo11 yolo26 yoloe26
```

## Release Packages

The exporter writes one self-contained zip per model under `releases/` and
updates `models.json`:

```text
releases/yolo26n-seg.zip
  model.onnx
  classes.names
```

The checked-in `models.json` intentionally stays small so OpenShot can populate
a compact dropdown such as `YOLO26: Nano`, `YOLOv11: Nano`, and `YOLOv8: Nano`.

## Details

All exports use:

- `640x640` static input
- batch size `1`
- ONNX opset `17`
- `dynamic=False`
- `nms=False`
- `end2end=False`
- `simplify=True`

The ONNX output is raw YOLO segmentation output. Consumers are responsible for
post-processing.

Label files are shared instead of duplicated per model:

```text
models/labels/coco80.names
models/labels/yoloe26-4585.names
models/labels/manifest.json
```

Release zips duplicate the required labels as `classes.names` so every download
is self-contained.

## Examples

```bash
python scripts/export_yolo_seg_onnx.py --sizes n
python scripts/export_yolo_seg_onnx.py --families yolo26
python scripts/export_yolo_seg_onnx.py --families yoloe26 --sizes n
python scripts/export_yolo_seg_onnx.py --force-download
python scripts/export_yolo_seg_onnx.py --skip-opencv-validate
```

## Links

- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [Ultralytics model assets](https://github.com/ultralytics/assets/releases)
- [YOLOv8 docs](https://docs.ultralytics.com/models/yolov8/)
- [YOLO11 docs](https://docs.ultralytics.com/models/yolo11/)
- [YOLO26 docs](https://docs.ultralytics.com/models/yolo26/)
- [YOLOE docs](https://docs.ultralytics.com/models/yoloe/)
- [Ultralytics license](https://www.ultralytics.com/license)
- [OpenShot](https://www.openshot.org/)
- [OpenShot GitHub](https://github.com/OpenShot/openshot-qt)
- [libopenshot](https://github.com/OpenShot/libopenshot)
- [OpenCV DNN](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html)

## Project Notes

This is an export utility, not an official Ultralytics or OpenShot project.
See [MODEL_ARTIFACTS.md](MODEL_ARTIFACTS.md) and [NOTICE.md](NOTICE.md) for
artifact and licensing notes. The repository code is MIT licensed; upstream
model weights and generated ONNX files remain subject to Ultralytics licensing.
