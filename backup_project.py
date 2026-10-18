#!/usr/bin/env python
"""Backup script: create a dated backup of the entire project."""
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKUP_ROOT = ROOT / "backups"
COMMIT = datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> int:
    dest = BACKUP_ROOT / COMMIT
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(ROOT, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tree_output.txt", "backups"))
    total = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(f"Backup created: {dest}")
    print(f"Files copied, total size: {total/1024/1024:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
