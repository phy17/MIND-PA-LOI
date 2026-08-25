#!/usr/bin/env python3
"""Build the task manifests for the two cloud sweeps (T-ITS revision-proofing).

E1  PA-LOI parameter ablation on the representative 27-scene set:
    six one-dimensional variants around the deployed operating point
    (eta 0.4/0.8, v_floor 1.0/3.0, w_max 15/35), each evaluated with
    (a) instant-center ghost probes at 5.5 m and 4.0 m triggers (650 frames)
    and (b) the no-ghost efficiency protocol (650 frames). The deployed
    point itself (eta=0.6, v_floor=2.0, w_max=25) is NOT rerun: its numbers
    already exist in the published clean81 tables.

E2  Moving-pedestrian (dash) protocol at additional trigger distances:
    4.5 m and 3.5 m, walking speeds 1.0/2.0/3.0 m/s, three primary stacks,
    800-frame horizon -- the same settings as the published 5.5 m protocol.

E3  Prepared reviewer-request extension (not part of the current paper):
    Reachable-set and Dynamic-shadow on the complete finite-speed walking
    matrix used by the three primary stacks: four speeds at 5.5 m and three
    speeds at 4.5/3.5 m, 27 scenes, 800-frame horizon (540 tasks). This closes
    the largest remaining baseline-validity gap if cloud time is available.

Scene set: the rep30 speed-sensitivity JSONL minus the three scenes excluded
by the paper (clean81 indices 23, 44, 70), i.e. the same 27 scenes behind
Table IX.

Output: <out>/records/*.jsonl (single-record scene files),
        <out>/tasks_e1.jsonl, <out>/tasks_e2.jsonl, and
        <out>/tasks_e3.jsonl (queue manifests).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXCLUDED_CLEAN81 = {23, 44, 70}

E1_VARIANTS = [
    ("eta0p4", {"PALOI_ETA": "0.4"}),
    ("eta0p8", {"PALOI_ETA": "0.8"}),
    ("vfloor1p0", {"PALOI_VFLOOR": "1.0"}),
    ("vfloor3p0", {"PALOI_VFLOOR": "3.0"}),
    ("wmax15", {"PALOI_WMAX": "15.0"}),
    ("wmax35", {"PALOI_WMAX": "35.0"}),
]
E1_DISTANCES = [5.5, 4.0]
E2_DISTANCES = [4.5, 3.5]
E2_SPEEDS = [1.0, 2.0, 3.0]
E2_BASELINES = ["ours", "aeb_only", "mind"]
E3_SPEEDS_BY_DISTANCE = {5.5: [1.0, 1.5, 2.0, 3.0], 4.5: [1.0, 2.0, 3.0], 3.5: [1.0, 2.0, 3.0]}
E3_BASELINES = ["reachset", "shadow"]


def tag(x: float) -> str:
    return ("%g" % float(x)).replace(".", "p")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path,
                    default=Path("数据集/ghost_injection_clean81_rep30_speed_sensitivity_d5p5_20260607.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("sweep_results"))
    ap.add_argument("--python", default="python3")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    scenes = [r for r in rows if int(r.get("_clean81_index", -1)) not in EXCLUDED_CLEAN81]
    assert len(scenes) == 27, f"expected 27 scenes, got {len(scenes)}"

    rec_dir = args.out / "records"
    rec_dir.mkdir(parents=True, exist_ok=True)
    rec_paths: dict[int, Path] = {}
    for r in scenes:
        idx = int(r["_clean81_index"])
        p = rec_dir / f"scene{idx:02d}.jsonl"
        p.write_text(json.dumps(r, ensure_ascii=False) + "\n", encoding="utf-8")
        rec_paths[idx] = p

    ghost_runner = "experiments/ghost_probe/run_jsonl_our_system_videos.py"
    noghost_runner = "experiments/ghost_probe/run_no_ghost_jsonl_efficiency.py"

    e1 = []
    for vname, venv in E1_VARIANTS:
        for idx, rec in sorted(rec_paths.items()):
            for d in E1_DISTANCES:
                tid = f"e1_{vname}_s{idx:02d}_d{tag(d)}"
                tdir = args.out / "e1" / vname / f"ghost_s{idx:02d}_d{tag(d)}"
                e1.append({
                    "task_id": tid, "task_dir": str(tdir),
                    "log_path": str(tdir) + ".log",
                    "env": venv,
                    "cmd": [args.python, "-u", ghost_runner,
                            "--jsonl", str(rec), "--select-mode", "first",
                            "--num-scenes", "1", "--output", str(tdir),
                            "--sim-horizon", "650", "--trigger-distance", str(d),
                            "--ghost-spawn-mode", "instant_center",
                            "--trigger-min-frame", "350", "--num-threads", "1",
                            "--baseline", "ours",
                            "--sim-name", f"ours_{vname}_d{tag(d)}_ic",
                            "--no-render"],
                })
            tid = f"e1_{vname}_s{idx:02d}_noghost"
            tdir = args.out / "e1" / vname / f"noghost_s{idx:02d}"
            e1.append({
                "task_id": tid, "task_dir": str(tdir),
                "log_path": str(tdir) + ".log",
                "env": venv,
                "cmd": [args.python, "-u", noghost_runner,
                        "--jsonl", str(rec), "--output", str(tdir),
                        "--baseline", "ours", "--sim-horizon", "650",
                        "--num-threads", "1",
                        "--sim-name", f"ours_{vname}_noghost"],
            })

    e2 = []
    for idx, rec in sorted(rec_paths.items()):
        for d in E2_DISTANCES:
            for v in E2_SPEEDS:
                for b in E2_BASELINES:
                    tid = f"e2_s{idx:02d}_d{tag(d)}_v{tag(v)}_{b}"
                    tdir = args.out / "e2" / f"s{idx:02d}_d{tag(d)}_v{tag(v)}_{b}"
                    e2.append({
                        "task_id": tid, "task_dir": str(tdir),
                        "log_path": str(tdir) + ".log",
                        "env": {},
                        "cmd": [args.python, "-u", ghost_runner,
                                "--jsonl", str(rec), "--select-mode", "first",
                                "--num-scenes", "1", "--output", str(tdir),
                                "--sim-horizon", "800", "--trigger-distance", str(d),
                                "--pedestrian-speed", str(v),
                                "--ghost-spawn-mode", "dash",
                                "--trigger-min-frame", "350", "--num-threads", "1",
                                "--baseline", b,
                                "--sim-name", f"{b}_dash_v{tag(v)}_d{tag(d)}",
                                "--no-render"],
                    })

    e3 = []
    for idx, rec in sorted(rec_paths.items()):
        for d, speeds in E3_SPEEDS_BY_DISTANCE.items():
            for v in speeds:
                for b in E3_BASELINES:
                    tid = f"e3_s{idx:02d}_d{tag(d)}_v{tag(v)}_{b}"
                    tdir = args.out / "e3" / f"s{idx:02d}_d{tag(d)}_v{tag(v)}_{b}"
                    e3.append({
                        "task_id": tid, "task_dir": str(tdir),
                        "log_path": str(tdir) + ".log",
                        "env": {},
                        "cmd": [args.python, "-u", ghost_runner,
                                "--jsonl", str(rec), "--select-mode", "first",
                                "--num-scenes", "1", "--output", str(tdir),
                                "--sim-horizon", "800", "--trigger-distance", str(d),
                                "--pedestrian-speed", str(v),
                                "--ghost-spawn-mode", "dash",
                                "--trigger-min-frame", "350", "--num-threads", "1",
                                "--baseline", b,
                                "--sim-name", f"{b}_dash_v{tag(v)}_d{tag(d)}",
                                "--no-render"],
                    })

    for name, tasks in [("tasks_e1.jsonl", e1), ("tasks_e2.jsonl", e2),
                        ("tasks_e3.jsonl", e3)]:
        path = args.out / name
        path.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks),
                        encoding="utf-8")
        print(f"{path}: {len(tasks)} tasks")


if __name__ == "__main__":
    main()
