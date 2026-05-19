# EfficientSAM ONNX Exports for OpenShot

This directory packages EfficientSAM ONNX models used by OpenShot's Object Mask
effect. EfficientSAM turns user prompts on seed frames into an initial object
mask; Cutie then propagates that mask through video.

The current OpenShot/libopenshot integration expects:

```text
image_segmentation_efficientsam_ti_2025april.onnx
```

Generated ONNX files and release zip packages are intentionally ignored by Git.

## Models

Two `1024x1024` variants are packaged:

```text
efficient-sam-tiny-1024.zip
  image_segmentation_efficientsam_ti_2025april.onnx

efficient-sam-small-static-1024.zip
  image_segmentation_efficientsam_s_static_1024.onnx
```

Use Tiny for fast Object Mask quality tiers and Small for high quality tiers.

### Tiny

- Published by OpenCV at `opencv/image_segmentation_efficientsam`.
- Apache-2.0 model-card license.
- Static `1024x1024` image input.
- Supports foreground points, background/filter points, and box prompts.
- Size: about 47 MB.
- SHA-256: `4eb496e0a7259d435b49b66faf1754aa45a5c382a34558ddda9a8c6fe5915d77`.

### Small

- Published upstream at `yunyangx/EfficientSAM`.
- Apache-2.0 model-card license.
- Source ONNX has dynamic image and point dimensions.
- Export script converts it to static `1024x1024` with six prompt slots for OpenCV DNN.
- Source size: about 102 MB.
- Source SHA-256: `b257787eeecdfd0db0626f83a8241874c35c74eb4c25c4d12ff0a478f90f30f9`.

OpenCV DNN inputs:

```text
batched_images       1x3x1024x1024 float32 RGB 0..1
batched_point_coords 1x1x6x2       float32, scaled to 1024x1024
batched_point_labels 1x1x6x1       float32
```

Prompt labels:

```text
1   foreground point
-1  background point/filter point
2   box top-left
3   box bottom-right
```

Outputs:

```text
output_masks     1x1x3x1024x1024
iou_predictions  1x1x3
```

## Build Release Package

From the repository root:

```bash
python efficient-sam/scripts/package_efficient_sam.py
```

The script:

1. Downloads the Tiny and Small ONNX models from Hugging Face.
2. Verifies the SHA-256 checksum.
3. Converts the upstream Small model to a static `1024x1024` OpenCV-friendly ONNX.
4. Compiles the C++ OpenCV DNN probe.
5. Validates a forward pass when sample data is available.
6. Writes zip assets and a manifest under `efficient-sam/releases/`.

The Small conversion requires:

```bash
python -m pip install -r efficient-sam/requirements.txt
```

Generated local directories:

```text
efficient-sam/models/     downloaded ONNX models, ignored by Git
efficient-sam/build/      compiled validation probe, ignored by Git
efficient-sam/releases/   generated release zip assets, ignored by Git
```

Skip validation:

```bash
python efficient-sam/scripts/package_efficient_sam.py --skip-validate
```

Build only one variant:

```bash
python efficient-sam/scripts/package_efficient_sam.py --variant tiny
python efficient-sam/scripts/package_efficient_sam.py --variant small
```

Reuse an existing model file:

```bash
mkdir -p efficient-sam/models
cp /path/to/image_segmentation_efficientsam_ti_2025april.onnx efficient-sam/models/
python efficient-sam/scripts/package_efficient_sam.py --skip-download
```

## Release Zip Contents

```text
efficient-sam-tiny-1024.zip
  image_segmentation_efficientsam_ti_2025april.onnx

efficient-sam-small-static-1024.zip
  image_segmentation_efficientsam_s_static_1024.onnx
```

The exporter also writes `efficient-sam/releases/efficient-sam-models.json`
with the asset name, checksum, model filename, and size.

## Probe

The validation probe can also be run directly:

```bash
g++ -std=c++17 efficient-sam/scripts/efficient_sam_opencv_probe.cpp \
  -o efficient-sam/build/efficient_sam_opencv_probe \
  $(pkg-config --cflags --libs opencv4)

efficient-sam/build/efficient_sam_opencv_probe \
  efficient-sam/models/image_segmentation_efficientsam_ti_2025april.onnx \
  experiments/xmem_onnx_temp/XMem_Export/sample/test-sample1-1frame.png \
  efficient-sam/releases/dog_point \
  570,1180,1
```

The probe writes:

```text
*_mask.png
*_overlay.png
```

## Sources

- OpenCV Hugging Face model card: https://huggingface.co/opencv/image_segmentation_efficientsam
- OpenCV sample wrapper: https://huggingface.co/opencv/image_segmentation_efficientsam/blob/main/efficientSAM.py
- Upstream EfficientSAM Hugging Face models: https://huggingface.co/yunyangx/EfficientSAM
- EfficientSAM upstream repository: https://github.com/yformer/EfficientSAM
- EfficientSAM paper: https://arxiv.org/abs/2312.00863

Review upstream licensing before redistributing weights or derived ONNX exports.
