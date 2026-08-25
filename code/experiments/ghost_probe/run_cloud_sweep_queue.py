#!/usr/bin/env python3
"""Generic manifest-driven task queue for the cloud parameter sweeps.

Reads a manifest JSONL in which every row fully describes one closed-loop
task (command, environment overrides, task directory), and executes the
pending rows with a bounded worker pool. Designed for a shared machine:

  - resume-safe: a task whose ``<task_dir>/summary.json`` exists is skipped;
  - polite: workers run under ``nice`` and OMP/BLAS threads are pinned to 1;
  - GPU round-robin: worker slot i sets CUDA_VISIBLE_DEVICES = i % --num-gpus;
  - memory gate: no new task is launched while available RAM is below the
    threshold, so the queue can never squeeze other users of the machine.

Manifest row schema:
  {"task_id": str, "task_dir": str, "log_path": str,
   "cmd": [str, ...], "env": {str: str}}

Usage:
  python3 run_cloud_sweep_queue.py --manifest sweep/tasks.jsonl \
      --max-workers 16 --num-gpus 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def task_done(task: dict) -> bool:
    p = Path(task["task_dir"]) / "summary.json"
    if not p.exists():
        return False
    try:
        return bool(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return False


def available_ram_gb() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return float("inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--num-gpus", type=int, default=0,
                    help="0 = leave CUDA_VISIBLE_DEVICES untouched")
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--min-free-ram-gb", type=float, default=20.0)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--poll-s", type=float, default=5.0)
    ap.add_argument("--launch-gap-s", type=float, default=0.5)
    args = ap.parse_args()

    tasks = read_manifest(args.manifest)
    state_dir = args.manifest.parent
    progress_path = state_dir / "progress.json"

    pending = [t for t in tasks if not task_done(t)]
    skipped = len(tasks) - len(pending)
    print(f"[QUEUE] total={len(tasks)} done_already={skipped} pending={len(pending)} "
          f"workers={args.max_workers} gpus={args.num_gpus}", flush=True)

    attempts: dict[str, int] = {}
    active: dict[str, tuple[subprocess.Popen, int, float]] = {}
    failed: list[str] = []
    completed = skipped
    slots = list(range(args.max_workers))
    queue = list(pending)

    def launch(task: dict, slot: int) -> subprocess.Popen:
        env = os.environ.copy()
        env.update({
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "THEANO_FLAGS": "blas.ldflags=",
        })
        env.update(task.get("env", {}))
        if args.num_gpus > 0:
            env["CUDA_VISIBLE_DEVICES"] = str(slot % args.num_gpus)
        Path(task["task_dir"]).mkdir(parents=True, exist_ok=True)
        log_path = Path(task["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "a", encoding="utf-8")
        log.write(f"\n[QUEUE START] {datetime.now().isoformat()} slot={slot} "
                  f"attempt={attempts.get(task['task_id'], 0) + 1}\n")
        log.flush()
        cmd = ["nice", "-n", str(args.nice)] + task["cmd"]
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)

    t0 = time.time()
    while queue or active:
        # Reap finished workers.
        for tid in list(active):
            proc, slot, started = active[tid]
            ret = proc.poll()
            if ret is None:
                continue
            del active[tid]
            slots.append(slot)
            task = next(t for t in tasks if t["task_id"] == tid)
            if ret == 0 and task_done(task):
                completed += 1
                print(f"[DONE] {tid} in {time.time() - started:.0f}s "
                      f"({completed}/{len(tasks)})", flush=True)
            else:
                attempts[tid] = attempts.get(tid, 0) + 1
                if attempts[tid] < args.max_attempts:
                    print(f"[RETRY] {tid} ret={ret} attempt={attempts[tid]}", flush=True)
                    queue.insert(0, task)
                else:
                    failed.append(tid)
                    print(f"[FAIL] {tid} ret={ret} giving up", flush=True)

        # Launch new tasks while capacity and memory allow.
        while queue and slots and available_ram_gb() >= args.min_free_ram_gb:
            task = queue.pop(0)
            if task_done(task):
                completed += 1
                continue
            slot = slots.pop(0)
            active[task["task_id"]] = (launch(task, slot), slot, time.time())
            time.sleep(args.launch_gap_s)

        progress_path.write_text(json.dumps({
            "ts": datetime.now().isoformat(),
            "total": len(tasks), "completed": completed,
            "active": len(active), "pending": len(queue),
            "failed": failed, "elapsed_s": round(time.time() - t0),
        }, indent=1), encoding="utf-8")
        time.sleep(args.poll_s)

    print(f"[QUEUE END] completed={completed}/{len(tasks)} failed={len(failed)} "
          f"elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
