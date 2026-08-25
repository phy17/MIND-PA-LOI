#!/usr/bin/env python3
"""Run one JSONL-selected AV2 scene without injecting any ghost pedestrian.

The scene record is still used to select the AV2 scenario, but the simulator
does not create a ghost agent. This measures traffic efficiency and incidental
collisions in the original traffic scene for a planner baseline.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("THEANO_FLAGS", "blas.ldflags=")

import planners.mind.planner as planner_module  # noqa: E402
from agent import CustomizedAgent, NonReactiveAgent  # noqa: E402
from common.geometry import check_polygon_intersection  # noqa: E402
from experiments.ghost_probe.run_ghost_experiment import GhostProbeSimulator  # noqa: E402


BASELINES = ("ours", "aeb_only", "mind", "reachset", "shadow")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scenario_id(rec: dict) -> str:
    seq = rec.get("seq_id") or rec.get("scenario_id")
    if not seq:
        raise ValueError(f"Record has no seq_id/scenario_id: {rec}")
    return seq


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


def select_record(rows: list[dict], args: argparse.Namespace) -> dict:
    if args.index is None:
        if len(rows) != 1:
            raise ValueError(f"{args.jsonl} has {len(rows)} records; pass --index or provide a one-row JSONL")
        return rows[0]
    for row in rows:
        if int(row.get("index", -1)) == args.index or int(row.get("clean21_index", -1)) == args.index:
            return row
    if 1 <= args.index <= len(rows):
        return rows[args.index - 1]
    raise ValueError(f"No record index {args.index} in {args.jsonl}")


def build_config(base_config: Path, rec: dict, out_dir: Path, args: argparse.Namespace) -> Path:
    seq = scenario_id(rec)
    cfg = json.loads(base_config.read_text(encoding="utf-8"))
    cfg["seq_id"] = seq
    cfg["sim_name"] = args.sim_name
    cfg["output_dir"] = str(out_dir) + "/"
    cfg["render"] = bool(args.render)
    cfg["num_threads"] = int(args.num_threads)
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


def write_frame_log(path: Path, frame_log: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step",
        "sim_time_s",
        "ego_x",
        "ego_y",
        "ego_vel_mps",
        "ego_heading_rad",
        "phase",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(frame_log)


def summarize_speeds(frame_log: list[dict], control_start_frame: int) -> dict:
    if not frame_log:
        return {
            "n_frames": 0,
            "avg_speed_all_mps": None,
            "avg_speed_control_mps": None,
            "min_speed_all_mps": None,
            "max_speed_all_mps": None,
            "final_speed_mps": None,
            "distance_all_m": 0.0,
            "distance_control_m": 0.0,
            "slow_pct_all_lt6": None,
            "slow_pct_control_lt6": None,
            "stop_pct_control_lt0p3": None,
        }

    speeds = np.array([float(row["ego_vel_mps"]) for row in frame_log], dtype=float)
    steps = np.array([int(row["step"]) for row in frame_log], dtype=int)
    xy = np.array([[float(row["ego_x"]), float(row["ego_y"])] for row in frame_log], dtype=float)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1) if len(xy) > 1 else np.array([], dtype=float)
    control_mask = steps >= control_start_frame
    control_speeds = speeds[control_mask]
    control_seg = seg[steps[:-1] >= control_start_frame] if len(seg) else np.array([], dtype=float)

    def pct(mask: np.ndarray) -> float | None:
        if mask.size == 0:
            return None
        return float(100.0 * np.mean(mask))

    return {
        "n_frames": int(len(frame_log)),
        "avg_speed_all_mps": float(np.mean(speeds)),
        "avg_speed_control_mps": float(np.mean(control_speeds)) if control_speeds.size else None,
        "min_speed_all_mps": float(np.min(speeds)),
        "max_speed_all_mps": float(np.max(speeds)),
        "final_speed_mps": float(speeds[-1]),
        "distance_all_m": float(np.sum(seg)) if len(seg) else 0.0,
        "distance_control_m": float(np.sum(control_seg)) if len(control_seg) else 0.0,
        "slow_pct_all_lt6": pct(speeds < 6.0),
        "slow_pct_control_lt6": pct(control_speeds < 6.0),
        "stop_pct_control_lt0p3": pct(control_speeds < 0.3),
    }


def run_no_ghost_sim(sim: GhostProbeSimulator, control_start_frame: int) -> tuple[list[dict], bool, int | None]:
    from tqdm import tqdm

    print("[NO_GHOST] Running closed-loop scene without ghost injection...", flush=True)
    sim.frames = []
    sim.sim_time = 0.0
    frame_log: list[dict] = []
    terminated = False
    planner_failure_frame: int | None = None
    collided = False

    for step_idx in tqdm(range(sim.sim_horizon)):
        frame = {}

        agent_obs = []
        for agent in sim.agents:
            if (isinstance(agent, NonReactiveAgent) and agent.is_valid()) or isinstance(agent, CustomizedAgent):
                agent_obs.append(agent.observe())

        agent_gt = []
        for agent in sim.agents:
            if (isinstance(agent, NonReactiveAgent) and agent.is_valid()) or isinstance(agent, CustomizedAgent):
                agent_gt.append(agent.observe_no_noise())
        frame["agents"] = agent_gt

        ego_agent = next((a for a in sim.agents if a.id == "AV"), None)
        if ego_agent is not None:
            ego_state = np.array(ego_agent.state, dtype=float)
            frame_log.append(
                {
                    "step": int(step_idx),
                    "sim_time_s": round(float(sim.sim_time), 4),
                    "ego_x": round(float(ego_state[0]), 5),
                    "ego_y": round(float(ego_state[1]), 5),
                    "ego_vel_mps": round(float(ego_state[2]), 5) if len(ego_state) > 2 else 0.0,
                    "ego_heading_rad": round(float(ego_state[3]), 6) if len(ego_state) > 3 else 0.0,
                    "phase": "closed_loop" if step_idx >= control_start_frame else "dataset_replay",
                }
            )

        if ego_agent and ego_agent.is_enable and not collided:
            ego_poly = sim.get_agent_polygon(ego_agent)
            for other in sim.agents:
                if other.id == "AV":
                    continue
                if np.linalg.norm(other.state[:2] - ego_agent.state[:2]) > 10.0:
                    continue
                other_poly = sim.get_agent_polygon(other)
                if check_polygon_intersection(ego_poly, other_poly):
                    print(
                        f"\n[COLLISION] At {sim.sim_time:.2f}s: Ego collided with {other.id} ({other.type})",
                        flush=True,
                    )
                    collided = True
                    sim.collision_log.append(
                        {
                            "timestamp": float(sim.sim_time),
                            "frame_idx": int(step_idx),
                            "ego_state": ego_agent.state.tolist(),
                            "ego_vel": float(ego_agent.state[2]),
                            "other_id": str(other.id),
                            "other_type": str(other.type),
                            "other_state": other.state.tolist(),
                            "collision_msg": f"Collision with {other.type} ID:{other.id}",
                        }
                    )

        for agent in sim.agents:
            if isinstance(agent, CustomizedAgent):
                agent.check_enable(sim.sim_time)
                rec_tri, pl_tri = agent.check_trigger(sim.sim_time)

                if rec_tri:
                    agent.step()
                if pl_tri:
                    agent.update_observation(agent_obs)
                    if agent.is_enable:
                        is_success, res = agent.plan()
                        if not is_success:
                            print(f"[NO_GHOST] Agent {agent.id} plan failed at step {step_idx}", flush=True)
                            if agent.id == "AV":
                                sim.collision_log.append(
                                    {
                                        "timestamp": float(sim.sim_time),
                                        "frame_idx": int(step_idx),
                                        "ego_state": ego_agent.state.tolist() if ego_agent else [],
                                        "ego_vel": float(ego_agent.state[2]) if ego_agent else 0.0,
                                        "other_id": "PLANNER_CRASH",
                                        "other_type": "ERROR",
                                        "other_state": [],
                                        "collision_msg": "Planner crashed (NaNs/Failed optimization).",
                                    }
                                )
                            planner_failure_frame = int(step_idx)
                            terminated = True
                            break

                        if agent.id == "AV":
                            frame["scen_tree"] = res[0]
                            frame["traj_tree"] = res[1]
                            if len(res) > 2:
                                frame["ghost_points"] = res[2]
                            if hasattr(agent, "gt_tgt_lane"):
                                frame["target_lane"] = agent.gt_tgt_lane

            elif isinstance(agent, NonReactiveAgent):
                agent.step()
            else:
                raise ValueError("Unknown agent type")
            agent.update_state(sim.sim_step)

        sim.frames.append(frame)
        if not sim.render and len(sim.frames) > 50:
            sim.frames = sim.frames[-10:]

        sim.sim_time += sim.sim_step
        if terminated:
            print("[NO_GHOST] Simulation terminated early.", flush=True)
            break

    print(
        f"[NO_GHOST] Complete. Ghost spawned: {sim.ghost_spawned}, collisions: {len(sim.collision_log)}",
        flush=True,
    )
    return frame_log, terminated, planner_failure_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one no-ghost AV2 efficiency scene.")
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--base-config", type=Path, default=Path("configs/ghost_experiment_vehicle.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", choices=BASELINES, default="ours")
    parser.add_argument("--sim-horizon", type=int, default=650)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--control-start-frame", type=int, default=None)
    parser.add_argument("--sim-name", default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--figsize", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--font-size", type=int, default=8)
    parser.add_argument("--data-logging", action="store_true")
    parser.add_argument("--debug-log", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.jsonl)
    rec = select_record(rows, args)
    seq = scenario_id(rec)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.sim_name is None:
        args.sim_name = f"{args.baseline}_no_ghost_efficiency"

    enable_ghost_probe, enable_aeb, baseline_mode = baseline_flags(args.baseline)
    importlib.reload(planner_module)
    planner_module.ENABLE_GHOST_PROBE = enable_ghost_probe
    planner_module.ENABLE_AEB = enable_aeb
    planner_module.BASELINE_MODE = baseline_mode
    planner_module.ENABLE_DATA_LOGGING = bool(args.data_logging)
    planner_module.DEBUG_LOG_ENABLED = bool(args.debug_log)

    config_path = build_config(args.base_config, rec, args.output, args)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.control_start_frame is None:
        enable_timestep = float(config["cl_agents"][0].get("enable_timestep", 4.0))
        args.control_start_frame = int(round(enable_timestep / 0.02))

    sim = GhostProbeSimulator(str(config_path), enable_ghost_probe_defense=enable_ghost_probe)
    sim.sim_horizon = int(args.sim_horizon)
    sim.render = bool(args.render)
    sim.plan_ambush = lambda: None

    t0 = time.perf_counter()
    sim.init_sim()
    sim.ghost_config = None
    sim.ghost_agent = None
    sim.ghost_spawned = False
    frame_log, terminated, planner_failure_frame = run_no_ghost_sim(sim, args.control_start_frame)
    elapsed_s = time.perf_counter() - t0

    frame_csv = args.output / "frame_log.csv"
    write_frame_log(frame_csv, frame_log)

    planner_log_path = None
    logger_collision_count = None
    logger_frame_count = None
    av_agent = next((a for a in sim.agents if a.id == "AV"), None)
    if av_agent and hasattr(av_agent, "planner") and hasattr(av_agent.planner, "save_experiment_log"):
        if getattr(av_agent.planner, "data_logger", None) is not None:
            av_agent.planner.data_logger.collision_count = len(sim.collision_log)
            logger_collision_count = int(getattr(av_agent.planner.data_logger, "collision_count", 0))
            logger_frame_count = int(getattr(av_agent.planner.data_logger, "frame_count", 0))
        planner_log_path = av_agent.planner.save_experiment_log()
        if getattr(av_agent.planner, "data_logger", None) is not None:
            av_agent.planner.data_logger = None

    sim.save_collision_report()
    if sim.render:
        sim.render_video()

    speed_summary = summarize_speeds(frame_log, args.control_start_frame)
    completed_full_horizon = len(frame_log) >= int(args.sim_horizon)
    fps_wall = len(frame_log) / elapsed_s if elapsed_s > 0 else math.nan
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment_type": "no_ghost_efficiency",
        "seq_id": seq,
        "jsonl": str(args.jsonl),
        "record_index": rec.get("index"),
        "clean21_index": rec.get("clean21_index"),
        "baseline": args.baseline,
        "enable_ghost_probe": enable_ghost_probe,
        "enable_aeb": enable_aeb,
        "baseline_mode": baseline_mode,
        "ghost_injected": False,
        "ghost_spawned": bool(sim.ghost_spawned),
        "sim_horizon": int(args.sim_horizon),
        "control_start_frame": int(args.control_start_frame),
        "completed_full_horizon": bool(completed_full_horizon),
        "terminated": bool(terminated),
        "planner_failure_frame": planner_failure_frame,
        "collision_count_ground_truth": len(sim.collision_log),
        "first_collision": sim.collision_log[0] if sim.collision_log else None,
        "collision_log": sim.collision_log,
        "planner_log_path": planner_log_path,
        "planner_logger_collision_count": logger_collision_count,
        "planner_logger_frame_count": logger_frame_count,
        "frame_log_csv": str(frame_csv),
        "elapsed_s": round(float(elapsed_s), 3),
        "wall_fps": round(float(fps_wall), 4) if not math.isnan(fps_wall) else None,
        **speed_summary,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[NO_GHOST] Summary saved: {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
