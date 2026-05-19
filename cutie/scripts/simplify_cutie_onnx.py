#!/usr/bin/env python3
# Copyright (c) 2026 OpenShot Studios, LLC
# SPDX-License-Identifier: MIT
"""Simplify Cutie ONNX slices and make attention-mask graphs OpenCV-friendly."""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxsim import simplify


def initializer_array(graph, name):
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def constant_array(node):
    for attr in node.attribute:
        if attr.name == "value" and attr.HasField("t"):
            return numpy_helper.to_array(attr.t)
    return None


def split_multi_axis_unsqueeze(model):
    graph = model.graph
    output_to_node = {output: node for node in graph.node for output in node.output}
    new_nodes = []
    changed = False

    for node in graph.node:
        if node.op_type != "Unsqueeze" or len(node.input) < 2:
            new_nodes.append(node)
            continue

        axes = initializer_array(graph, node.input[1])
        constant_node = output_to_node.get(node.input[1])
        if axes is None and constant_node and constant_node.op_type == "Constant":
            axes = constant_array(constant_node)
        if axes is None:
            new_nodes.append(node)
            continue

        axes = [int(axis) for axis in np.asarray(axes).flatten().tolist()]
        if len(axes) <= 1:
            new_nodes.append(node)
            continue

        changed = True
        current = node.input[0]
        for index, axis in enumerate(axes):
            axis_name = f"{node.name or node.output[0]}_axis_{index}"
            axis_tensor = helper.make_tensor(axis_name, TensorProto.INT64, [1], [axis])
            graph.initializer.append(axis_tensor)
            output = node.output[0] if index == len(axes) - 1 else f"{node.output[0]}_unsqueeze_{index}"
            new_nodes.append(helper.make_node(
                "Unsqueeze",
                [current, axis_name],
                [output],
                name=f"{node.name or node.output[0]}_split_{index}",
            ))
            current = output

    if changed:
        del graph.node[:]
        graph.node.extend(new_nodes)
    return changed


def simplify_model(src, output):
    model = onnx.load(src)
    sim, ok = simplify(model)
    if not ok:
        raise RuntimeError(f"simplify failed: {src}")
    onnx.save(sim, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--opencv-unsqueeze", action="store_true")
    args = parser.parse_args()

    for src in args.models:
        sim_path = src.with_name(src.stem + "-sim.onnx")
        simplify_model(src, sim_path)
        print(f"saved {sim_path}")

        if args.opencv_unsqueeze:
            model = onnx.load(sim_path)
            split_multi_axis_unsqueeze(model)
            opencv_path = src.with_name(src.stem + "-opencv.onnx")
            onnx.save(model, opencv_path)
            print(f"saved {opencv_path}")


if __name__ == "__main__":
    main()
