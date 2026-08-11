"""Build a clean, shareable zip of the node pack.

The working folder contains development caches and a `workflow/` directory full
of large test images. Zipping the folder as-is produces a ~300 MB download for
something that should be under a megabyte. This script copies only what a user
needs.

    python scripts/make_release_zip.py            -> releases/ComfyUI-Smart-Upscaler-v1.0.0.zip
    python scripts/make_release_zip.py --out DIR  -> somewhere else

The zip unpacks to a single `ComfyUI-Smart-Upscaler/` folder, which is exactly
what someone drops into `ComfyUI/custom_nodes/`.
"""

import argparse
import hashlib
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "ComfyUI-Smart-Upscaler"

# Everything a user needs, and nothing else.
INCLUDE_FILES = (
    "__init__.py",
    "preset_store.py",
    "pyproject.toml",
    "LICENSE",
    "README.md",
    "CHANGELOG.md",
)
INCLUDE_DIRS = ("nodes", "web", "presets", "docs", "tests", "scripts")

# Workflows are named explicitly. `workflow/` is also the working directory for
# test images and parked graphs from other projects; none of that ships.
SHIP_WORKFLOWS = (
    "Smart-Upscaler-Z-Turbo-v1a.json",
    "Upscale_Test_Image.png",
)

SKIP_DIR_NAMES = {"__pycache__", ".git", ".claude", ".agents", "legacy", "releases"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".log"}


def version():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else "0.0.0"


def copy_tree(source, target):
    for item in sorted(source.rglob("*")):
        if any(part in SKIP_DIR_NAMES for part in item.parts):
            continue
        if item.is_dir() or item.suffix in SKIP_SUFFIXES:
            continue
        destination = target / item.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination)


def build(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{PACKAGE}-v{version()}"
    archive = out_dir / f"{name}.zip"

    with tempfile.TemporaryDirectory() as temporary:
        staged = Path(temporary) / PACKAGE
        staged.mkdir(parents=True)

        for filename in INCLUDE_FILES:
            source = ROOT / filename
            if source.exists():
                shutil.copy2(source, staged / filename)

        for dirname in INCLUDE_DIRS:
            source = ROOT / dirname
            if source.is_dir():
                copy_tree(source, staged / dirname)

        workflows = staged / "workflow"
        workflows.mkdir(parents=True, exist_ok=True)
        shipped = []
        for filename in SHIP_WORKFLOWS:
            source = ROOT / "workflow" / filename
            if not source.exists():
                raise SystemExit(f"Missing workflow to ship: {source}")
            shutil.copy2(source, workflows / filename)
            shipped.append(source)

        if archive.exists():
            archive.unlink()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for item in sorted(staged.rglob("*")):
                if item.is_file():
                    bundle.write(item, item.relative_to(staged.parent))

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (out_dir / f"{name}.zip.sha256").write_text(
        f"{digest}  {archive.name}\n", encoding="utf-8"
    )

    with zipfile.ZipFile(archive) as bundle:
        count = len(bundle.namelist())
    size = archive.stat().st_size / (1024 * 1024)
    print(f"{archive}")
    print(f"  {count} files, {size:.2f} MB")
    print(f"  sha256 {digest[:16]}...")
    print(f"  workflows: {', '.join(item.name for item in shipped) or 'none'}")
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "releases"))
    build(Path(parser.parse_args().out))
