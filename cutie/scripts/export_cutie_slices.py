#!/usr/bin/env python3
# Copyright (c) 2026 OpenShot Studios, LLC
# SPDX-License-Identifier: MIT
"""Export fixed-size Cutie network slices to ONNX for OpenCV DNN."""

import argparse
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir

SCRIPT_DIR = Path(__file__).resolve().parent
CUTIE_DIR = SCRIPT_DIR.parent
DEFAULT_CUTIE_ROOT = CUTIE_DIR / "vendor" / "Cutie"
DEFAULT_WEIGHTS = CUTIE_DIR / "weights" / "cutie-base-mega.pth"
DEFAULT_OUTPUT_DIR = CUTIE_DIR / "models"


def load_cutie_modules(cutie_root):
    """Load official Cutie modules from a local checkout."""
    sys.path.insert(0, str(cutie_root))
    global CUTIE, aggregate, cutie_modules, do_softmax, get_similarity, readout
    from cutie.model.cutie import CUTIE
    from cutie.model.utils.memory_utils import do_softmax, get_similarity, readout
    from cutie.utils.tensor_utils import aggregate
    import cutie.model.modules as cutie_modules


class EncodeKeyWrapper(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, image):
        ms_features, pix_feat = self.network.encode_image(image)
        key, shrinkage, selection = self.network.transform_key(ms_features[0])
        return ms_features[0], ms_features[1], ms_features[2], pix_feat, key, shrinkage, selection


class EncodeValueWrapper(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, image, pix_feat, sensory, mask):
        mask_value, new_sensory, object_memory, _object_logits = self.network.encode_mask(
            image,
            pix_feat,
            sensory,
            mask,
            deep_update=True,
            chunk_size=-1,
            need_weights=False,
        )
        return mask_value, new_sensory, object_memory


class DecodeWrapper(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, f16, f8, f4, memory_readout, sensory):
        new_sensory, logits, prob = self.network.segment(
            [f16, f8, f4],
            memory_readout,
            sensory,
            chunk_size=-1,
            update_sensory=True,
        )
        return new_sensory, logits, prob


class MemoryReadoutWrapper(torch.nn.Module):
    def __init__(self, network, top_k):
        super().__init__()
        self.network = network
        self.top_k = top_k

    def visual_readout(self, query_key, query_selection, memory_key, memory_shrinkage, memory_value, memory_valid):
        batch_size, num_objects = memory_value.shape[:2]
        with torch.cuda.amp.autocast(enabled=False):
            similarity = get_similarity(
                memory_key.float(),
                memory_shrinkage.float(),
                query_key.float(),
                query_selection.float(),
            )
            valid = memory_valid.float().flatten(start_dim=1).unsqueeze(2)
            similarity = similarity + (1.0 - valid) * -10000.0
            affinity = do_softmax(similarity, top_k=self.top_k, inplace=False)
            flat_value = memory_value.flatten(start_dim=1, end_dim=2).float()
            pixel_readout = readout(affinity, flat_value)
            pixel_readout = pixel_readout.view(
                batch_size,
                num_objects,
                self.network.value_dim,
                *pixel_readout.shape[-2:],
            )
        return pixel_readout

    def forward(
        self,
        query_key,
        query_selection,
        memory_key,
        memory_shrinkage,
        memory_value,
        memory_valid,
        object_memory,
        pix_feat,
        sensory,
        last_mask,
    ):
        pixel_readout = self.visual_readout(
            query_key,
            query_selection,
            memory_key,
            memory_shrinkage,
            memory_value,
            memory_valid,
        )
        pixel_readout = self.network.pixel_fusion(pix_feat, pixel_readout, sensory, last_mask)
        memory_readout, _ = self.network.readout_query(pixel_readout, object_memory)
        return memory_readout


class BasicMemoryReadoutWrapper(MemoryReadoutWrapper):
    def forward(
        self,
        query_key,
        query_selection,
        memory_key,
        memory_shrinkage,
        memory_value,
        memory_valid,
        pix_feat,
        sensory,
        last_mask,
    ):
        pixel_readout = self.visual_readout(
            query_key,
            query_selection,
            memory_key,
            memory_shrinkage,
            memory_value,
            memory_valid,
        )
        return self.network.pixel_fusion(pix_feat, pixel_readout, sensory, last_mask)


def _static_interpolate_groups(g, ratio, mode, align_corners):
    batch_size, num_objects, channels, height, width = g.shape
    out_height = int(height * ratio)
    out_width = int(width * ratio)
    flat = g.reshape(batch_size * num_objects, channels, height, width)
    resized = F.interpolate(flat, size=(out_height, out_width), mode=mode, align_corners=align_corners)
    return resized.reshape(batch_size, num_objects, channels, out_height, out_width)


def _static_upsample_groups(g, ratio=2, mode="bilinear", align_corners=False):
    return _static_interpolate_groups(g, ratio, mode, align_corners)


def _static_downsample_groups(g, ratio=1 / 2, mode="area", align_corners=None):
    return _static_interpolate_groups(g, ratio, mode, align_corners)


def patch_static_group_interpolation():
    cutie_modules.upsample_groups = _static_upsample_groups
    cutie_modules.downsample_groups = _static_downsample_groups


def patch_object_transformer_attention_mask(network):
    if not getattr(network, "object_transformer_enabled", False):
        return

    def _no_aux_mask(self, logits, selector):
        return None

    network.object_transformer._get_aux_mask = types.MethodType(
        _no_aux_mask,
        network.object_transformer,
    )


def patch_object_transformer_float_attention_mask(network):
    if not getattr(network, "object_transformer_enabled", False):
        return

    def _float_aux_mask(self, logits, selector):
        if selector is None:
            prob = logits.sigmoid()
        else:
            prob = logits.sigmoid() * selector
        logits_with_bg = aggregate(prob, dim=1)

        is_foreground = logits_with_bg[:, 1:] >= logits_with_bg.max(dim=1, keepdim=True)[0]
        foreground_mask = is_foreground.float().flatten(start_dim=2)
        inv_foreground_mask = 1.0 - foreground_mask
        inv_background_mask = foreground_mask

        aux_foreground_mask = inv_foreground_mask.unsqueeze(2).unsqueeze(2).repeat(
            1, 1, self.num_heads, self.num_queries // 2, 1).flatten(start_dim=0, end_dim=2)
        aux_background_mask = inv_background_mask.unsqueeze(2).unsqueeze(2).repeat(
            1, 1, self.num_heads, self.num_queries // 2, 1).flatten(start_dim=0, end_dim=2)

        aux_mask = torch.cat([aux_foreground_mask, aux_background_mask], dim=1)
        fully_blocked = aux_mask.sum(-1, keepdim=True) >= aux_mask.shape[-1]
        aux_mask = aux_mask * (1.0 - fully_blocked.float())
        return aux_mask * -10000.0

    network.object_transformer._get_aux_mask = types.MethodType(
        _float_aux_mask,
        network.object_transformer,
    )


def export_model(model, inputs, output_path, input_names, output_names, opset):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutie-root", type=Path, default=DEFAULT_CUTIE_ROOT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--memory-frames", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    args.cutie_root = args.cutie_root.expanduser().resolve()
    args.weights = args.weights.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    load_cutie_modules(args.cutie_root)
    initialize_config_dir(
        version_base="1.3.2",
        config_dir=str(args.cutie_root / "cutie" / "config"),
        job_name="cutie_onnx_export",
    )
    cfg = compose(config_name="eval_config")
    network = CUTIE(cfg).eval()
    network.load_weights(torch.load(args.weights, map_location="cpu"))
    patch_static_group_interpolation()

    width = int(args.width or args.size)
    height = int(args.height or args.size)
    if width % 16 != 0 or height % 16 != 0:
        raise ValueError("--width and --height must be divisible by 16")
    size_name = str(width) if width == height else f"{width}x{height}"

    image = torch.rand(1, 3, height, width, dtype=torch.float32)
    with torch.no_grad():
        ms_features, pix_feat = network.encode_image(image)
        key, _, _ = network.transform_key(ms_features[0])

    objects = 1
    h16 = key.shape[-2]
    w16 = key.shape[-1]
    sensory = torch.zeros(1, objects, cfg.model.sensory_dim, h16, w16, dtype=torch.float32)
    mask = torch.rand(1, objects, height, width, dtype=torch.float32)
    memory_readout = torch.rand(1, objects, cfg.model.embed_dim, h16, w16, dtype=torch.float32)
    memory_key = torch.rand(1, cfg.model.key_dim, args.memory_frames, h16, w16, dtype=torch.float32)
    memory_shrinkage = torch.rand(1, 1, args.memory_frames, h16, w16, dtype=torch.float32) + 1.0
    memory_value = torch.rand(
        1,
        objects,
        cfg.model.value_dim,
        args.memory_frames,
        h16,
        w16,
        dtype=torch.float32,
    )
    memory_valid = torch.ones(1, 1, args.memory_frames, h16, w16, dtype=torch.float32)
    object_memory = torch.rand(
        1,
        objects,
        1,
        cfg.model.object_summarizer.num_summaries,
        cfg.model.embed_dim + 1,
        dtype=torch.float32,
    )
    last_mask = torch.rand(1, objects, height, width, dtype=torch.float32)

    export_model(
        EncodeKeyWrapper(network),
        image,
        args.output_dir / f"cutie-encode-key-{size_name}.onnx",
        ["image"],
        ["f16", "f8", "f4", "pix_feat", "key", "shrinkage", "selection"],
        args.opset,
    )
    export_model(
        EncodeValueWrapper(network),
        (image, pix_feat, sensory, mask),
        args.output_dir / f"cutie-encode-value-{size_name}.onnx",
        ["image", "pix_feat", "sensory", "mask"],
        ["mask_value", "new_sensory", "object_memory"],
        args.opset,
    )
    export_model(
        DecodeWrapper(network),
        (ms_features[0], ms_features[1], ms_features[2], memory_readout, sensory),
        args.output_dir / f"cutie-decode-{size_name}.onnx",
        ["f16", "f8", "f4", "memory_readout", "sensory"],
        ["new_sensory", "logits", "prob"],
        args.opset,
    )
    patch_object_transformer_attention_mask(network)
    export_model(
        MemoryReadoutWrapper(network, args.top_k),
        (
            key,
            torch.rand_like(key),
            memory_key,
            memory_shrinkage,
            memory_value,
            memory_valid,
            object_memory,
            pix_feat,
            sensory,
            last_mask,
        ),
        args.output_dir / f"cutie-memory-readout-nomask-valid-{size_name}-m{args.memory_frames}-topk{args.top_k}.onnx",
        [
            "query_key",
            "query_selection",
            "memory_key",
            "memory_shrinkage",
            "memory_value",
            "memory_valid",
            "object_memory",
            "pix_feat",
            "sensory",
            "last_mask",
        ],
        ["memory_readout"],
        args.opset,
    )
    export_model(
        BasicMemoryReadoutWrapper(network, args.top_k),
        (
            key,
            torch.rand_like(key),
            memory_key,
            memory_shrinkage,
            memory_value,
            memory_valid,
            pix_feat,
            sensory,
            last_mask,
        ),
        args.output_dir / f"cutie-memory-readout-basic-valid-{size_name}-m{args.memory_frames}-topk{args.top_k}.onnx",
        [
            "query_key",
            "query_selection",
            "memory_key",
            "memory_shrinkage",
            "memory_value",
            "memory_valid",
            "pix_feat",
            "sensory",
            "last_mask",
        ],
        ["memory_readout"],
        args.opset,
    )

    network = CUTIE(cfg).eval()
    network.load_weights(torch.load(args.weights, map_location="cpu"))
    patch_static_group_interpolation()
    patch_object_transformer_float_attention_mask(network)
    export_model(
        MemoryReadoutWrapper(network, args.top_k),
        (
            key,
            torch.rand_like(key),
            memory_key,
            memory_shrinkage,
            memory_value,
            memory_valid,
            object_memory,
            pix_feat,
            sensory,
            last_mask,
        ),
        args.output_dir / f"cutie-memory-readout-floatmask-valid-{size_name}-m{args.memory_frames}-topk{args.top_k}.onnx",
        [
            "query_key",
            "query_selection",
            "memory_key",
            "memory_shrinkage",
            "memory_value",
            "memory_valid",
            "object_memory",
            "pix_feat",
            "sensory",
            "last_mask",
        ],
        ["memory_readout"],
        args.opset,
    )

    print(f"Exported Cutie ONNX slices to {args.output_dir}")


if __name__ == "__main__":
    main()
