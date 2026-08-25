#!/usr/bin/env python3
"""Single-scene mechanism figure for the T-ITS manuscript.

Reads the per-planning-frame logger CSVs and run summaries produced by

    experiments/ghost_probe/run_jsonl_our_system_videos.py --data-logging

for one clean81 scene under the three primary stacks (ours / aeb_only / mind),
and renders speed-versus-remaining-distance traces that show *how* PA-LOI
reshapes the approach before the ghost trigger while the reference stacks
arrive fast and collide.

Default inputs are the scene61 (seq d567a51c, clean81 index 61) reruns in
output/mechanism_case/, executed with the exact main-protocol settings
(trigger 5.5 m, instant_center, 650 frames, trigger-min-frame 350).

Usage:
    python3 paper/tits_pa_loi/scripts/figure_mechanism_case.py \
        [--case-dir output/mechanism_case] [--out paper/tits_pa_loi/figures]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]

STACKS = [
    ("mind", "MIND", "#9ca3af", "-"),
    ("aeb_only", "AEB-only", "#d97706", "--"),
    ("ours", "PA-LOI + AEB", "#0f766e", "-"),
]
LOW_SEV = 3.0  # m/s reporting threshold


def load_logger_csv(run_dir: Path) -> dict[str, np.ndarray]:
    logs = sorted(run_dir.glob("scene_*/logs/log_*.csv"))
    if not logs:
        raise FileNotFoundError(f"no logger csv under {run_dir}")
    rows = list(csv.DictReader(open(logs[0], encoding="utf-8")))
    def col(name: str) -> np.ndarray:
        return np.array([float(r[name]) for r in rows], dtype=float)
    v = col("Ego_Vel")
    vel_factor = col("Vel_Factor")  # hinge excess squared, [v - v_safe]_+^2
    active = (col("Risk_Source_Dist") > 0) & (col("Risk_Cost_Total") > 0)
    # Reconstruct the active safe-speed bound from the logged hinge excess.
    v_safe = np.where(vel_factor > 1e-9, v - np.sqrt(np.maximum(vel_factor, 0.0)), np.nan)
    return {
        "x": col("Ego_X"), "y": col("Ego_Y"), "v": v,
        "v_req": v_safe, "active": active.astype(float),
        "frame": col("Frame"),
    }


def load_summary(run_dir: Path) -> dict:
    """Run summary; empty dict while the simulation is still in progress."""
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data[0] if isinstance(data, list) else data


def remaining_distance(xs: np.ndarray, ys: np.ndarray, conflict: np.ndarray) -> np.ndarray:
    """Path-aligned remaining distance to the conflict point.

    Arc length is accumulated along the logged ego positions; the conflict
    arc coordinate is taken at the point of closest approach, so the same
    definition serves runs that stop short and runs that drive through.
    """
    pts = np.stack([xs, ys], axis=1)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    d2 = np.linalg.norm(pts - conflict[None, :], axis=1)
    i_min = int(np.argmin(d2))
    # If the run ends before reaching the conflict, extend by the straight-line gap.
    s_c = s[i_min] + (d2[i_min] if d2[i_min] > 0.05 else 0.0)
    return s_c - s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", type=Path, default=REPO / "output/mechanism_case")
    ap.add_argument("--out", type=Path, default=REPO / "paper/tits_pa_loi/figures")
    args = ap.parse_args()

    record = json.loads((args.case_dir / "scene61_record.jsonl").read_text(encoding="utf-8"))
    conflict = np.array(record["target_pos"], dtype=float)
    trigger_d = 5.5

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)

    annots = []
    for key, label, color, ls in STACKS:
        run_dir = args.case_dir / key
        log = load_logger_csv(run_dir)
        summ = load_summary(run_dir)
        d = remaining_distance(log["x"], log["y"], conflict)
        v = log["v"]
        first = summ.get("first_collision") or {}
        imp = float(first.get("ego_vel", 0.0)) if first else 0.0
        collided = summ.get("collision_count", 0) > 0
        # Truncate a colliding trace at the conflict point; post-impact frames
        # are not informative for the approach mechanism.
        if collided:
            reach = np.nonzero(d <= 0.05)[0]
            end = int(reach[0]) + 1 if len(reach) else len(d)
        else:
            end = len(d)
        lw = 2.0 if key == "mind" else 1.5
        ax.plot(d[:end], v[:end], ls, color=color, lw=lw, label=label,
                solid_capstyle="round")
        if collided:
            annots.append((key, label, color, imp))
            ax.plot([0.0], [imp], "x", color=color, ms=6.5, mew=1.9, zorder=6)

        if key == "ours":
            act = log["active"] > 0.5
            if act.any():
                # PA-LOI safe-speed bound while the phantom source is active.
                ax.plot(d[act], log["v_req"][act], "-", color=color,
                        lw=0.8, alpha=0.5)
                d_act = d[act]
                ax.axvspan(max(float(d_act.min()), 0.0), float(d_act.max()),
                           color=color, alpha=0.05, lw=0)

    ax.set_ylim(-0.3, 10.6)
    ax.axvline(trigger_d, color="#b91c1c", lw=0.9, ls="-.")
    ax.text(trigger_d + 0.7, 9.1, "ghost trigger (5.5 m)",
            color="#b91c1c", fontsize=6.5, va="top", ha="right", rotation=90)
    ax.axhline(LOW_SEV, color="0.55", lw=0.7, ls=":")
    ax.text(13.5, LOW_SEV - 0.75, "low-severity threshold (3 m/s)",
            color="0.4", fontsize=6, ha="left")
    ax.set_xlim(-1.0, 38.0)
    ax.invert_xaxis()
    ax.set_xlabel("remaining distance to conflict point (m)")
    ax.set_ylabel("ego speed (m/s)")
    ax.legend(loc="lower left", framealpha=0.95)
    ax.grid(alpha=0.25, lw=0.4)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "mechanism_case.pdf"
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    for key, label, color, imp in annots:
        print(f"  {label}: ghost impact at {imp:.2f} m/s")


if __name__ == "__main__":
    main()
