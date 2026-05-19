# YOLO ONNX Exports for OpenShot

This directory builds OpenCV-friendly ONNX exports of official Ultralytics YOLO
segmentation models for use in OpenShot and libopenshot.

The generated models are static ONNX graphs: no Python, PyTorch, or Ultralytics
runtime is needed after export. OpenShot/libopenshot can load them with OpenCV
DNN and handle thresholds, NMS, labels, boxes, and masks in C++.

## Getting Started

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps -r yolo/requirements.txt

python yolo/scripts/export_yolo_seg_onnx.py
```

Default output goes here:

```text
yolo/models/pt/       downloaded official checkpoints
yolo/models/onnx/     exported ONNX models
yolo/models/labels/   shared label files and manifest
yolo/releases/        generated GitHub Release zip assets
yolo/models.json      friendly release/download catalog for consumers
```

Large model binaries and generated release zips are ignored by Git. `models.json`
is the file apps and scripts should read: it lists the friendly model names,
release zip assets, and download base URL. During release packaging, the exporter
refreshes each generated zip checksum.

## Models

Default export:

```bash
python yolo/scripts/export_yolo_seg_onnx.py
```

Exports nano, small, and medium segmentation models for:

- `yolov8` COCO 80 classes
- `yolo11` COCO 80 classes
- `yolo26` COCO 80 classes

Optional broad-vocabulary YOLOE export:

```bash
python yolo/scripts/export_yolo_seg_onnx.py --families yoloe26
```

YOLOE-26 uses the official prompt-free checkpoint vocabulary, embeds its 4,585
labels with MobileCLIP during export, and writes a static ONNX model. MobileCLIP
is not needed at runtime.

Export everything:

```bash
python yolo/scripts/export_yolo_seg_onnx.py --families yolov8 yolo11 yolo26 yoloe26
```

## Release Packages

The exporter writes one self-contained zip per model under `yolo/releases/` and
updates `yolo/models.json`:

```text
yolo/releases/yolo26n-seg.zip
  model.onnx
  classes.names
```

The checked-in `yolo/models.json` intentionally stays small so OpenShot can
populate a compact dropdown such as `YOLO26: Nano`, `YOLOv11: Nano`, and
`YOLOv8: Nano`.

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
yolo/models/labels/coco80.names
yolo/models/labels/yoloe26-4585.names
yolo/models/labels/manifest.json
```

Release zips duplicate the required label file inside each zip as
`classes.names`.

## Examples

```bash
python yolo/scripts/export_yolo_seg_onnx.py --sizes n
python yolo/scripts/export_yolo_seg_onnx.py --families yolo26
python yolo/scripts/export_yolo_seg_onnx.py --families yoloe26 --sizes n
python yolo/scripts/export_yolo_seg_onnx.py --force-download
python yolo/scripts/export_yolo_seg_onnx.py --skip-opencv-validate
```

## Links

- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [Ultralytics model assets](https://github.com/ultralytics/assets/releases)
- [YOLOv8 docs](https://docs.ultralytics.com/models/yolov8/)
- [YOLO11 docs](https://docs.ultralytics.com/models/yolo11/)
- [YOLO26 docs](https://docs.ultralytics.com/models/yolo26/)
- [YOLOE docs](https://docs.ultralytics.com/models/yoloe/)
- [Ultralytics license](https://www.ultralytics.com/license)
