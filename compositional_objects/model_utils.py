import json
import os
import shutil
from pathlib import Path
from typing import List, Optional


def open_select_glb_file_dialog() -> str:
    """Create a simple GUI for picking a GLB file."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file = filedialog.askopenfilename(
        title="Select GLB file", filetypes=[("GLB files", "*.glb")]
    )
    return file


class HabitatSceneDataset:
    """Utility class for importing GLB files from Blender into a Habitat.

    Most the action takes place in `import_model`. There's a bit of housekeeping
    to generate filenames, but that's about it.
    """

    # - Habitat-based file naming conventions.
    _attrs_suffix = ".object_config.json"
    _attrs_digits = 3
    _attrs_sep = "_"

    def __init__(self, dataset_dir: os.PathLike):
        self.dataset_dir = Path(dataset_dir).expanduser()
        if not self.dataset_dir.is_dir():
            raise NotADirectoryError(
                f"Dataset directory '{dataset_dir}' is not a directory."
            )
        self.mesh_dir = self.dataset_dir / "meshes"
        self.config_dir = self.dataset_dir / "configs"

    @classmethod
    def create(
        cls, path: os.PathLike, name: Optional[str] = None
    ) -> "HabitatSceneDataset":
        """Create a new Habitat dataset.

        Args:
            path: Path to the dataset directory. Must not already exist.
            name: Name of the dataset. This is only used to name the scene
              attributes file (e.g. "myobject.scene_dataset_config.json").
              If None (the default), it'll use the parent directory's name.

        Raises:
            FileExistsError: If the dataset path already exists.

        Returns:
            HabitatSceneDataset: The new dataset.
        """

        # Figure out the directory path and dataset name.
        path = Path(path).expanduser()
        if path.exists():
            raise FileExistsError(f"Dataset path '{path}' already exists.")
        name = name or path.name

        # Create the directory structure.
        path.mkdir(parents=True)
        for dir_name in ["meshes", "configs"]:
            (path / dir_name).mkdir()

        # Create scene dataset attributes.
        scene_attrs = {"objects": {"paths": {".json": ["configs/"]}}}
        scene_attrs_path = path / f"{name}.scene_dataset_config.json"
        with open(scene_attrs_path, "w") as f:
            json.dump(scene_attrs, f, indent=2)

        return cls(path)

    def list_objects(self) -> List[dict]:
        infos = []
        for config_path in self.config_dir.glob(f"*{self._attrs_suffix}"):
            uid = config_path.name[: -len(self._attrs_suffix)]
            num = int(uid[: self._attrs_digits])
            name = uid[self._attrs_digits + len(self._attrs_sep) :]
            infos.append(
                {
                    "uid": uid,
                    "num": num,
                    "name": name,
                }
            )
        return infos

    def import_model(
        self,
        src: Optional[os.PathLike] = None,
        name: Optional[str] = None,
        replace: bool = False,
    ) -> None:
        """Import a GLB model into the dataset.

        Args:
            src: Path to the source GLB file. If None, opens a file dialog.
            name: Name for the imported object. If None, uses the source filename.
            replace: If True, replaces existing object with same name. If False,
                raises FileExistsError if object exists.

        Raises:
            ValueError: If select file dialog is cancelled.
            FileExistsError: If an object with same name exists and `replace=False`.
        """
        if src is None:
            src = open_select_glb_file_dialog()
            if not src:
                raise ValueError("No GLB file selected.")
        src = Path(src)

        # - Determine model name, a model number (for the file prefix), and
        #   its unique ID (e.g., '001_tbp_mug').
        name = name or src.stem
        all_objects = self.list_objects()
        existing = next((obj for obj in all_objects if obj["name"] == name), None)
        if existing is None:
            if len(all_objects) > 0:
                num = max([obj["num"] for obj in all_objects]) + 1
            else:
                num = 1
            info = {
                "uid": f"{num:0{self._attrs_digits}d}{self._attrs_sep}{name}",
                "num": num,
                "name": name,
            }
        else:
            if not replace:
                raise FileExistsError(f"Object '{name}' already exists.")
            info = existing

        # - Write object attributes file.
        attrs = {
            "friction_coefficient": 3.0,
            "render_asset": None,
            "requires_lighting": True,
            "up": [0.0, 1.0, 0.0],
            "front": [0.0, 1.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        }

        attrs["render_asset"] = f"../meshes/{info['uid']}/textured.glb"

        with open(self.config_dir / f"{info['uid']}.object_config.json", "w") as f:
            json.dump(attrs, f, indent=2)

        # - Copy GLB file.
        dst = self.mesh_dir / f"{info['uid']}" / "textured.glb"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.unlink(missing_ok=True)
        shutil.copyfile(src, dst)


if __name__ == "__main__":
    blender_dir = Path.home() / "Google Drive/My Drive/Blender/compositional_objects"
    object_names = [
        "mug_tbp",
        "cube_tbp",
        "cylinder_tbp",
        "disk_tbp",
        "sphere_tbp",
    ]
    dataset_path = (
        Path(os.environ.get("MONTY_DATA", "~/tbp/data")).expanduser()
        / "compositional_objects"
    )
    if dataset_path.exists():
        dset = HabitatSceneDataset(dataset_path)
    else:
        dset = HabitatSceneDataset.create(dataset_path)

    dset.import_model(
        blender_dir / "templates/cylinder/cylinder_alpha_test.glb", replace=True
    )
