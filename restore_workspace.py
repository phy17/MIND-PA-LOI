#!/usr/bin/env python3
"""Restore this organized archive into the paths expected by the scripts."""
from __future__ import annotations
import argparse
import csv
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--destination", required=True, type=Path,
                    help="Root of a clean MIND checkout or empty workspace")
    args = ap.parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with (HERE / "RESTORE_MAP.tsv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    for row in rows:
        src = HERE / row["archive_path"]
        dst = destination / row["repository_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(f"Restored {len(rows)} files into {destination}")

if __name__ == "__main__":
    main()
