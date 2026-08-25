#!/usr/bin/env python3
"""Run JSONL-driven ghost-probe experiments for selected planner baselines."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("THEANO_FLAGS", "blas.ldflags=")

import planners.mind.planner as planner_module  # noqa: E402
from experiments.ghost_probe.render_ghost_scenes import (  # noqa: E402
    candidate_from_dict,
    render_scene,
)
from experiments.ghost_probe.run_ghost_experiment import GhostProbeSimulator  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scenario_id(rec: dict) -> str:
    seq = rec.get("seq_id") or rec.get("scenario_id")
    if not seq:
        raise ValueError(f"Ghost injection record has no seq_id/scenario_id: {rec}")
    return seq


def load_manifest_indices(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    return {rec["seq_id"]: int(rec["index"]) for rec in records}


def build_config(base_config: Path, rec: dict, out_dir: Path, args: argparse.Namespace) -> Path:
    seq = scenario_id(rec)
    cfg = json.loads(base_config.read_text(encoding="utf-8"))
    cfg["seq_id"] = seq
    cfg["sim_name"] = args.sim_name
    cfg["output_dir"] = str(out_dir) + "/"
    cfg["render"] = args.render
    cfg["num_threads"] = args.num_threads
    cfg["use_cuda"] = False
    cfg["render_config"] = {
        "mode": "follow",
        "camera_position": {"x": 0, "y": 0, "yaw": 0, "elev": 90},
        "figsize": args.figsize,
        "dpi": args.dpi,
        "font_size": args.font_size,
    }

    config_dir = out_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{seq}.json"
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return config_path


def render_overview_if_possible(rec: dict, data_dir: Path, out_path: Path, label: str) -> str | None:
    try:
        candidate = candidate_from_dict(rec)
    except TypeError:
        return None
    render_scene(candidate, data_dir, out_path, title_prefix=f"[{label}] ")
    return str(out_path)


def baseline_flags(name: str) -> tuple[bool, bool, str | None]:
    """Map baseline name -> (enable_ghost_probe, enable_aeb, baseline_mode).

    reachset/shadow are published-baseline supervisors (planner.BASELINE_MODE):
    PA-LOI off, AEB on (config parity), supervisor clamps planner output.
    """
    if name == "ours":
        return True, True, None
    if name == "aeb_only":
        return False, True, None
    if name == "mind":
        return False, False, None
    if name == "reachset":
        return False, True, "reachset"
    if name == "shadow":
        return False, True, "shadow"
    raise ValueError(f"Unknown baseline: {name}")


def run_one(rec: dict, scene_label: str, out_base: Path, args: argparse.Namespace) -> dict:
    seq = scenario_id(rec)
    enable_ghost_probe, enable_aeb, baseline_mode = baseline_flags(args.baseline)
    run_dir = out_base / f"{scene_label}_{seq[:8]}_{args.baseline}"
    run_dir.mkdir(parents=True, exist_ok=True)

    importlib.reload(planner_module)
    planner_module.ENABLE_GHOST_PROBE = enable_ghost_probe
    planner_module.ENABLE_AEB = enable_aeb
    planner_module.BASELINE_MODE = baseline_mode
    planner_module.ENABLE_DATA_LOGGING = args.data_logging
    planner_module.DEBUG_LOG_ENABLED = args.debug_log

    config_path = build_config(args.base_config, rec, run_dir, args)
    sim = GhostProbeSimulator(
        str(config_path),
        enable_ghost_probe_defense=enable_ghost_probe,
        ghost_injection_spec=rec,
        pedestrian_speed=args.pedestrian_speed,
        strict_trigger_dist=args.trigger_distance,
        trigger_mode=args.trigger_mode,
        trigger_min_frame=args.trigger_min_frame,
        trigger_max_frame=args.trigger_max_frame,
        ghost_spawn_mode=args.ghost_spawn_mode,
    )
    sim.sim_horizon = args.sim_horizon
    sim.render = args.render

    start = time.time()
    sim.run()
    elapsed = time.time() - start

    video_path = run_dir / f"{seq}_{args.sim_name}.mov"
    overview_path = None
    if args.render_overview:
        overview_path = render_overview_if_possible(
            rec,
            args.data_dir,
            out_base / "overview_pngs" / f"{scene_label}_{seq[:8]}.png",
            scene_label,
        )

    return {
        "scene_label": scene_label,
        "seq_id": seq,
        "output_dir": str(run_dir),
        "video_path": str(video_path),
        "overview_png": overview_path,
        "baseline": args.baseline,
        "enable_ghost_probe": enable_ghost_probe,
        "enable_aeb": enable_aeb,
        "baseline_mode": baseline_mode,
        "ghost_spawned": bool(sim.ghost_spawned),
        "ghost_spawn_mode": sim.ghost_spawn_mode,
        "collision_count": len(sim.collision_log),
        "collision_log": sim.collision_log,
        "first_collision": sim.collision_log[0] if sim.collision_log else None,
        "trigger_distance_m": float(getattr(sim, "strict_trigger_dist", rec.get("trigger_distance_m", 4.5))),
        "trigger_mode": sim.trigger_mode,
        "reference_trigger_frame": getattr(sim, "_reference_trigger_frame", None),
        "reference_trigger_frame_raw": getattr(sim, "_reference_trigger_frame_raw", None),
        "ghost_spawn_time_s": getattr(sim, "_ghost_spawn_time", None),
        "pedestrian_speed_ms": float(sim.pedestrian_speed),
        "elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JSONL-driven ghost-probe experiments.")
    parser.add_argument("--jsonl", type=Path, default=Path("数据集/ghost_injection_22_early650_full.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("数据集/manifest.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--base-config", type=Path, default=Path("configs/ghost_experiment_vehicle.json"))
    parser.add_argument("--output", type=Path, default=Path("output/jsonl_our_system_videos"))
    parser.add_argument("--num-scenes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--select-mode", choices=["random", "first", "indices"], default="random")
    parser.add_argument("--indices", nargs="*", type=int,
                        help="Dataset/JSONL indices to run when --select-mode=indices.")
    parser.add_argument("--sim-horizon", type=int, default=650)
    parser.add_argument("--trigger-mode", choices=["distance", "reference_time", "reference_frame", "scheduled"],
                        default="distance")
    parser.add_argument("--trigger-min-frame", type=int, default=220)
    parser.add_argument("--trigger-max-frame", type=int, default=None,
                        help="Default is sim_horizon - 50, leaving one second after trigger.")
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--figsize", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--font-size", type=int, default=8)
    parser.add_argument("--sim-name", default="our_system_jsonl")
    parser.add_argument("--baseline", choices=["ours", "aeb_only", "mind", "reachset", "shadow"], default="ours",
                        help="Planner baseline: ours=PA-LOI+AEB, aeb_only=AEB without PA-LOI, mind=plain MIND, "
                             "reachset=B1 reachable-set velocity bound, shadow=B2 dynamic shadow tracking.")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--render-overview", action="store_true",
                        help="Render a static overview PNG for each selected scene.")
    parser.add_argument("--trigger-distance", type=float, default=4.5)
    parser.add_argument("--pedestrian-speed", type=float, default=None)
    parser.add_argument("--ghost-spawn-mode", choices=["dash", "instant_center"], default="dash")
    parser.add_argument("--data-logging", action="store_true")
    parser.add_argument("--debug-log", action="store_true")
    args = parser.parse_args()
    args.render = not args.no_render
    if args.trigger_max_frame is None:
        args.trigger_max_frame = args.sim_horizon - 50

    rows = load_jsonl(args.jsonl)
    if args.num_scenes > len(rows):
        raise ValueError(f"Asked for {args.num_scenes} scenes but {args.jsonl} has {len(rows)}")

    if args.select_mode == "first":
        selected = rows[: args.num_scenes]
    elif args.select_mode == "indices":
        if not args.indices:
            raise ValueError("--indices is required when --select-mode=indices")
        wanted = set(args.indices)
        selected = [rec for rec in rows if int(rec.get("index", rows.index(rec) + 1)) in wanted]
        found = {int(rec.get("index", rows.index(rec) + 1)) for rec in selected}
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(f"Missing requested JSONL indices: {missing}")
    else:
        rng = random.Random(args.seed)
        selected = rng.sample(rows, args.num_scenes)
    manifest_indices = load_manifest_indices(args.manifest)

    args.output.mkdir(parents=True, exist_ok=True)
    selected_path = args.output / "selected_records.jsonl"
    selected_path.write_text(
        "".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in selected),
        encoding="utf-8",
    )

    print(f"[SMOKE] JSONL: {args.jsonl}")
    print(f"[SMOKE] Seed: {args.seed}")
    print("[SMOKE] Selected scenes:")
    for rec in selected:
        seq = scenario_id(rec)
        idx = manifest_indices.get(seq, "?")
        print(
            f"  - dataset_index={idx} seq_id={seq} cross_time={rec.get('cross_time_s')} "
            f"spawn_mode={args.ghost_spawn_mode}"
        )

    results = []
    for rec in selected:
        seq = scenario_id(rec)
        idx = manifest_indices.get(seq)
        label = f"scene{idx:02d}" if idx is not None else f"scene_{seq[:8]}"
        results.append(run_one(rec, label, args.output, args))

    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SMOKE] Summary: {summary_path}")
    for result in results:
        print(f"[SMOKE] Video: {result['video_path']}")
        if result["overview_png"]:
            print(f"[SMOKE] Overview: {result['overview_png']}")


if __name__ == "__main__":
    main()
