#!/usr/bin/env python3
"""Verify every file listed in MANIFEST.tsv against its SHA-256 digest."""
from __future__ import annotations
import csv
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

def main() -> None:
    with (HERE / "MANIFEST.tsv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    for row in rows:
        path = HERE / row["path"]
        assert path.is_file(), f"missing: {row['path']}"
        assert path.stat().st_size == int(row["bytes"]), f"size mismatch: {row['path']}"
        assert digest(path) == row["sha256"], f"digest mismatch: {row['path']}"
    print(f"Archive verification: PASS ({len(rows)} files)")

if __name__ == "__main__":
    main()
