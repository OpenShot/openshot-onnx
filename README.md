# OpenShot ONNX Exports

This repository builds OpenCV-friendly ONNX model packages for OpenShot and
libopenshot. The generated ONNX files and zip packages are static runtime
assets: OpenShot should not need Python, PyTorch, or the upstream training
frameworks after export.

## Model Families

| Family | Purpose | Directory |
| --- | --- | --- |
| YOLO | Object detection and instance segmentation | [`yolo/`](yolo/) |
| EfficientSAM | Prompted seed-mask generation for Object Mask | [`efficient-sam/`](efficient-sam/) |
| Cutie | Video object mask propagation for Object Mask | [`cutie/`](cutie/) |

XMem remains in `experiments/` as historical scratch work and is not promoted as
a supported release family.

## Release Artifacts

Generated model binaries are ignored by Git. Build them locally and upload the
zip files as GitHub Release assets after reviewing upstream model licenses.

Common outputs:

```text
yolo/models.json          YOLO release catalog
yolo/releases/            YOLO zip packages
efficient-sam/models.json EfficientSAM release catalog
efficient-sam/releases/   EfficientSAM zip packages
cutie/models.json         Cutie release catalog
cutie/releases/           Cutie zip packages
```

## Quick Commands

YOLO:

```bash
python yolo/scripts/export_yolo_seg_onnx.py
```

Cutie:

```bash
python cutie/scripts/export_cutie_quality_tiers.py
```

EfficientSAM:

```bash
python efficient-sam/scripts/package_efficient_sam.py
```

## Links

- [OpenShot](https://www.openshot.org/)
- [OpenShot GitHub](https://github.com/OpenShot/openshot-qt)
- [libopenshot](https://github.com/OpenShot/libopenshot)
- [OpenCV DNN](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html)

## Project Notes

This is an export utility, not an official upstream model project. See
[`MODELS.md`](MODELS.md) and [`NOTICE.md`](NOTICE.md) for
artifact and licensing notes. The repository code is MIT licensed; upstream
model weights and generated ONNX files remain subject to their upstream terms.
