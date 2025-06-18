import json
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from deepdiff import DeepDiff
from pygltflib import GLTF2

MONTY_DATA_DIR = Path(os.environ.get("MONTY_DATA", "~/tbp/data")).expanduser()

NEW_DATASET_NAME = "compositional_objects"
NEW_DATASET_DIR = MONTY_DATA_DIR / NEW_DATASET_NAME
assert NEW_DATASET_NAME == NEW_DATASET_DIR.name


"""
================================================================================

"""


@dataclass
class ObjectInfo:
    name: str
    num: int
    id: str
    config: dict = field(default_factory=dict)
    glb_path: Path = field(default_factory=Path)
    config_path: Path = field(default_factory=Path)


def pick_glb_file():
    """Create a simple GUI for picking two GLB files to compare."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file = filedialog.askopenfilename(
        title="Select GLB file", filetypes=[("GLB files", "*.glb")]
    )
    return file


class ObjectDataset:
    config_suffix = ".object_config.json"
    num_string_width = 3

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.meshes_dir = dataset_dir / "meshes"
        self.configs_dir = dataset_dir / "configs"

    def list_objects(self) -> List[ObjectInfo]:
        objects = []
        for p in self.configs_dir.glob(f"*{self.config_suffix}"):
            filename = p.name[: -len(self.config_suffix)]
            num_string = filename[: self.num_string_width]
            name = filename[self.num_string_width + 1 :]
            object_id = f"{int(num_string):0{self.num_string_width}d}_{name}"
            objects.append(
                ObjectInfo(
                    name=name,
                    num=int(num_string),
                    id=object_id,
                    config_path=p,
                    glb_path=p.parent.parent / "meshes" / object_id / "textured.glb",
                )
            )
        return objects

    def get_object(self, name: str) -> Optional[ObjectInfo]:
        for obj in self.list_objects():
            if obj.name == name:
                return obj
        return None

    def delete_object(self, name: str) -> Optional[ObjectInfo]:
        obj = self.get_object(name)
        if obj is None:
            return None

        obj.config_path.unlink(missing_ok=True)
        obj.glb_path.unlink(missing_ok=True)
        if obj.glb_path.parent.is_dir():
            obj.glb_path.parent.rmdir()
        return obj

    def import_blender_object(
        self,
        src_glb: Optional[os.PathLike] = None,
        object_name: Optional[str] = None,
        config: Optional[dict] = None,
        replace: bool = False,
    ) -> None:
        if src_glb is None:
            src_glb = pick_glb_file()
        src_glb = Path(src_glb)

        for dir_path in [self.dataset_dir, self.meshes_dir, self.configs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # - Pick Name, Number, and ID
        if object_name is None:
            object_name = src_glb.stem

        all_objects = self.list_objects()
        existing_object = self.get_object(object_name)
        if existing_object is None:
            if len(all_objects) > 0:
                object_num = max([obj.num for obj in all_objects]) + 1
            else:
                object_num = 1
        else:
            if not replace:
                print(f"Object '{object_name}' already exists. Quitting.")
                return
            print(f"Replacing existing object '{object_name}'.")
            object_num = existing_object.num
            self.delete_object(object_name)

        object_id = f"{object_num:0{self.num_string_width}d}_{object_name}"

        # - Copy model
        dst_glb = self.meshes_dir / f"{object_id}" / "textured.glb"
        dst_glb.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_glb, dst_glb)

        # - Config Template
        default_config = {
            "friction_coefficient": 3.0,
            "join_collision_meshes": False,
            "collision_asset": None,
            "render_asset": None,
            "requires_lighting": True,
            "up": [0.0, 1.0, 0.0],
            "front": [0.0, 1.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        }

        object_config = default_config.copy()
        if config:
            object_config.update(config)
        object_config["render_asset"] = f"../meshes/{object_id}/textured.glb"

        # Finally, write the mesh and config files.
        # - Copy model
        dst_glb = self.meshes_dir / f"{object_id}" / "textured.glb"
        dst_glb.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_glb, dst_glb)
        # - Write config
        with open(self.configs_dir / f"{object_id}{self.config_suffix}", "w") as f:
            json.dump(object_config, f, indent=2)


# ycb_dir = MONTY_DATA_DIR / "habitat" / "versioned_data" / "ycb_1.2"
# co_dir = NEW_DATASET_DIR


# collision_mesh_path = ycb_dir / "collison_meshes" / "010_potted_meat_cv_decomp.glb"
# config_path = ycb_dir / "configs" / "010_potted_meat_can.object_config.json"
# mesh_1_path = ycb_dir / "meshes" / "010_potted_meat_can" / "google_16k" / "textured.glb"
# mesh_2_path = (
#     ycb_dir / "meshes" / "010_potted_meat_can" / "google_16k" / "textured.glb.orig"
# )

# mug_path = co_dir / "tbp_mug.glb"


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

"""
================================================================================

"""
blender_dir = Path.home() / "Google Drive/My Drive/Blender"
in_path = blender_dir / "tbp_mug.glb"


dset = ObjectDataset(NEW_DATASET_DIR)
dset.import_blender_object(in_path, replace=True)
