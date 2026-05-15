# Model Artifacts

Generated model files are intentionally ignored by Git:

- `models/pt/*.pt`
- `models/onnx/*.onnx`
- `models/text/*.ts`
- `releases/*.zip`

The ONNX files are reproducible with:

```bash
python scripts/export_yolo_seg_onnx.py
```

They are not committed by default because pretrained YOLO checkpoints and
derived ONNX exports are licensed by Ultralytics, and some generated files are
large enough to require Git LFS or GitHub Releases instead of normal Git
storage.

The exporter code in this repository is MIT licensed. That license does not
change the license terms of Ultralytics checkpoints, MobileCLIP weights, or
derived model artifacts.

Supported outputs:

| Family | PT source release | ONNX files |
| --- | --- | --- |
| YOLO26 | `ultralytics/assets` `v8.4.0` | `yolo26n-seg.onnx`, `yolo26s-seg.onnx`, `yolo26m-seg.onnx` |
| YOLO11 | `ultralytics/assets` `v8.3.0` | `yolo11n-seg.onnx`, `yolo11s-seg.onnx`, `yolo11m-seg.onnx` |
| YOLOv8 | `ultralytics/assets` `v8.4.0` | `yolov8n-seg.onnx`, `yolov8s-seg.onnx`, `yolov8m-seg.onnx` |
| YOLOE-26 | `ultralytics/assets` `v8.4.0` plus prompt-free vocabulary checkpoints | `yoloe-26n-seg.onnx`, `yoloe-26s-seg.onnx`, `yoloe-26m-seg.onnx` |

If you want to publish generated ONNX files with a GitHub repository, use Git
LFS or attach them to a GitHub Release after reviewing Ultralytics licensing
requirements for your intended use.

Class labels are not duplicated per model. Shared label sets live in
`models/labels/`, and `models/labels/manifest.json` maps each ONNX filename to
the correct label file and output width.

Release packages duplicate the required label file inside each zip as
`classes.names`:

```text
yolo26n-seg.zip
  model.onnx
  classes.names
```

The root `models.json` file is the consumer-facing release catalog. It stays
small on purpose: friendly dropdown names, zip asset filenames, and release
download information. The exporter refreshes zip checksums when release packages
are generated.
