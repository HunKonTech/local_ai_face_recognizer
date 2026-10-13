#!/usr/bin/env python3
"""Build a distributable desktop bundle with PyInstaller."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from PyInstaller.__main__ import run as pyinstaller_run
from PyInstaller.utils.hooks import collect_data_files, collect_submodules


APP_NAME = "Face-Local"
ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build" / "pyinstaller"
DIST_DIR = ROOT / "dist"
ENTRYPOINT = ROOT / "app" / "main.py"
ASSETS_DIR = ROOT / "assets"
ICON_PNG = ASSETS_DIR / "icons" / "app-icon.png"
ICON_ICNS = ASSETS_DIR / "icons" / "face-local.icns"
ICON_ICO = ASSETS_DIR / "icons" / "face-local.ico"
# Keep in sync with app.diagnostics.HAAR_CASCADES (what the app loads).
HAAR_CASCADES = [
    "haarcascade_frontalface_default.xml",
    "haarcascade_profileface.xml",
    "haarcascade_eye.xml",
    "haarcascade_eye_tree_eyeglasses.xml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Display version for generated bundles.")
    return parser.parse_args()


def build() -> None:
    separator = ";" if os.name == "nt" else ":"
    data_args: list[str] = []

    for source, destination in iter_data_files():
        data_args.extend(["--add-data", f"{source}{separator}{destination}"])

    hidden_imports = sorted(set(collect_submodules("app")))

    # TFLite backends use dynamic imports that PyInstaller cannot auto-detect.
    # Add all three backends so the frozen app can find whichever is installed.
    for _tflite_mod in [
        "ai_edge_litert",
        "ai_edge_litert.interpreter",
        "tflite_runtime",
        "tflite_runtime.interpreter",
        "tensorflow",
        "tensorflow.lite",
    ]:
        if _tflite_mod not in hidden_imports:
            hidden_imports.append(_tflite_mod)

    # InsightFace (optional, the strongest detector of the false-positive
    # gate) is imported lazily; bundle its model-zoo loaders when the build
    # environment has it (`pip install .[insightface]`).  The buffalo_l model
    # pack itself is downloaded on first use, not bundled.
    if _installed("insightface"):
        for mod in collect_submodules(
            "insightface", filter=lambda name: _insightface_module_wanted(name)
        ):
            if mod not in hidden_imports:
                hidden_imports.append(mod)
    else:
        print("WARNING: insightface not installed - the bundle will lack the "
              "InsightFace detector (pip install .[insightface])")

    args = [
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name",
        APP_NAME,
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(BUILD_DIR),
        "--specpath",
        str(BUILD_DIR),
        "--paths",
        str(ROOT),
        *data_args,
    ]

    icon_path = build_icon_path()
    if icon_path is not None:
        args.extend(["--icon", str(icon_path)])

    if os.name == "posix" and os.uname().sysname == "Darwin":
        args.extend(["--osx-bundle-identifier", "local.face.recognizer"])

    for hidden_import in hidden_imports:
        args.extend(["--hidden-import", hidden_import])

    args.append(str(ENTRYPOINT))
    pyinstaller_run(args)


def iter_data_files() -> list[tuple[str, str]]:
    data_files: list[tuple[str, str]] = []
    for filename in ("config.example.yaml",):
        path = ROOT / filename
        if path.exists():
            data_files.append((str(path), "."))

    if ICON_PNG.exists():
        data_files.append((str(ICON_PNG), "assets/icons"))

    # OpenCV's Haar cascades (face / profile / eye) are data files the cv2
    # PyInstaller hook does not collect; without them the multi-stage
    # false-positive gate silently loses its Haar and eye technologies.
    data_files.extend(collect_data_files("cv2", subdir="data", includes=HAAR_CASCADES))

    models_dir = ROOT / "models"
    if models_dir.exists():
        for model_file in sorted(models_dir.iterdir()):
            if model_file.is_file():
                data_files.append((str(model_file), "models"))

    return data_files


def _installed(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


def _insightface_module_wanted(name: str) -> bool:
    """Skip InsightFace's GUI, CLI and training helpers (heavy optional deps)."""
    skipped = (
        "insightface.gui",
        "insightface.commands",
        "insightface.data.rec_builder",
        "insightface.thirdparty",
        "insightface.app.mask_renderer",
    )
    return not name.startswith(skipped)


def build_icon_path() -> Path | None:
    """Return the platform-appropriate icon file for PyInstaller."""
    if sys.platform == "darwin":
        return ICON_ICNS if ICON_ICNS.exists() else None
    if os.name == "nt":
        return ICON_ICO if ICON_ICO.exists() else None
    return ICON_PNG if ICON_PNG.exists() else None


def clean() -> None:
    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    shutil.rmtree(DIST_DIR / APP_NAME, ignore_errors=True)
    shutil.rmtree(DIST_DIR / f"{APP_NAME}.app", ignore_errors=True)


def inject_version(version: str) -> None:
    """Overwrite app/__init__.py with the build-time version."""
    init_path = ROOT / "app" / "__init__.py"
    init_path.write_text(
        f'"""Face-Local: offline face grouping and person labeling application."""\n\n'
        f'__version__ = "{version}"\n',
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    inject_version(args.version)
    clean()
    build()


if __name__ == "__main__":
    main()
