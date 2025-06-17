import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from deepdiff import DeepDiff
from pygltflib import GLTF2

DATASET_DIR = Path(os.environ["MONTY_DATA"]) / "compositional_objects"
ycb_dir = Path("/Users/sknudstrup/tbp/data/habitat/versioned_data/ycb_1.2")
co_dir = Path("/Users/sknudstrup/tbp/data/compositional_objects")


collision_mesh_path = ycb_dir / "collison_meshes" / "010_potted_meat_cv_decomp.glb"
config_path = ycb_dir / "configs" / "010_potted_meat_can.object_config.json"
mesh_1_path = ycb_dir / "meshes" / "010_potted_meat_can" / "google_16k" / "textured.glb"
mesh_2_path = (
    ycb_dir / "meshes" / "010_potted_meat_can" / "google_16k" / "textured.glb.orig"
)

mug_path = co_dir / "tbp_mug.glb"


def load_glb_summary(path):
    gltf = GLTF2().load_binary(path)
    return {
        "mesh_count": len(gltf.meshes or []),
        "material_count": len(gltf.materials or []),
        "node_count": len(gltf.nodes or []),
        "animation_count": len(gltf.animations or []),
        "texture_count": len(gltf.textures or []),
        "accessor_count": len(gltf.accessors or []),
        "buffer_view_count": len(gltf.bufferViews or []),
        "scene_count": len(gltf.scenes or []),
        "extensions_used": gltf.extensionsUsed or [],
        "extensions_required": gltf.extensionsRequired or [],
    }


def detailed_structure(path):
    gltf = GLTF2().load_binary(path)
    # Convert GLTF2 object into a JSON-compatible dict
    return gltf.model_dump(mode="json")


def print_summary(summary, label):
    print(f"\n📦 Summary for {label}:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def compare_glbs(file1, file2):
    summary1 = load_glb_summary(file1)
    summary2 = load_glb_summary(file2)

    print_summary(summary1, file1)
    print_summary(summary2, file2)

    print("\n🔍 Summary Differences:")
    diff_summary = DeepDiff(summary1, summary2, ignore_order=True)
    print(diff_summary.pretty() or "None")

    print("\n📊 Structural Differences (full model):")
    struct1 = detailed_structure(file1)
    struct2 = detailed_structure(file2)

    diff_struct = DeepDiff(struct1, struct2, ignore_order=True, verbose_level=1)
    print(diff_struct.pretty() or "None")
