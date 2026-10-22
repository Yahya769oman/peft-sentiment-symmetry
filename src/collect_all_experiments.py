from pathlib import Path
import zipfile
import platform
import sys
import subprocess
from datetime import datetime

PROJECT = Path(__file__).resolve().parent.parent
SRC = PROJECT / "src"
OUTPUTS = PROJECT / "outputs"
ZIP_PATH = PROJECT / "project_evidence_bundle.zip"
INFO_PATH = PROJECT / "experiment_environment_info.txt"

ALLOWED = {".py", ".json", ".csv", ".txt", ".log", ".md", ".png", ".pdf"}

def include_file(p):
    s = str(p)
    if "__pycache__" in s or ".ipynb_checkpoints" in s or "checkpoint-" in s:
        return False
    return p.is_file() and p.suffix.lower() in ALLOWED

def version(pkg):
    try:
        return subprocess.check_output(
            [sys.executable, "-c", f"import {pkg}; print({pkg}.__version__)"],
            text=True
        ).strip()
    except Exception:
        return "unknown"

info = [
    f"Created: {datetime.now().isoformat()}",
    f"Python: {platform.python_version()}",
    f"PyTorch: {version('torch')}",
    f"Transformers: {version('transformers')}",
    f"PEFT: {version('peft')}",
]

try:
    import torch
    info.append(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        info.append(f"GPU: {torch.cuda.get_device_name(0)}")
        info.append(f"GPU total MB: {torch.cuda.get_device_properties(0).total_memory/(1024**2):.3f}")
except Exception as e:
    info.append(f"GPU info error: {e!r}")

INFO_PATH.write_text("\n".join(info), encoding="utf-8")

files = []
for root in [SRC, OUTPUTS]:
    if root.exists():
        for p in root.rglob("*"):
            if include_file(p):
                files.append(p)

files.append(INFO_PATH)

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(set(files)):
        z.write(p, p.relative_to(PROJECT))

print("=" * 70)
print("EXPERIMENT PACKAGE CREATED")
print("=" * 70)
print(f"Files included: {len(set(files))}")
print(f"ZIP: {ZIP_PATH}")
print(f"Size MB: {ZIP_PATH.stat().st_size/(1024**2):.2f}")
