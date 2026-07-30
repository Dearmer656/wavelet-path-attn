#!/usr/bin/env python
"""Watch `squeue` every N seconds (default 1 hour) and estimate typical job durations.

What it does:
  1. Every ``--interval`` seconds, snapshots all currently RUNNING jobs
     cluster-wide (job id, job name, elapsed time, node, gres).
  2. Compares against the previous snapshot by job id. Any job id that was
     running last time but is gone now is treated as finished; its last-seen
     elapsed time is recorded as an estimated duration for jobs with that name.
  3. Estimates are appended (not overwritten) to a per-name history, so a
     recurring job name (e.g. the same training script run repeatedly)
     accumulates multiple duration samples over time.

Why track by job id internally but key estimates by job name: several
concurrently-running jobs can share the exact same name (common on a shared
cluster), so job id is the only reliable way to detect "this specific job
finished" -- but the whole point of this tool is "if I see a new job whose
name looks like one I've seen before, how long does it usually run", which
is a per-name question, so the recorded estimates are grouped by name.

Usage:
  python3 watch_squeue_job_durations.py [--interval 3600] [--state-dir DIR]

Run this in the background yourself, e.g.:
  nohup python3 watch_squeue_job_durations.py > watch_squeue.log 2>&1 &
or inside a screen/tmux session. It runs forever (Ctrl-C / kill to stop) and
is safe to kill and restart -- it reloads its state from disk each time.

Output files (under --state-dir, default ./squeue_job_duration_watch/):
  - current_state.json     : {job_id: {name, elapsed_seconds, node, gres, first_seen, last_seen}}
    for the most recent snapshot (used to detect disappearances on restart).
  - duration_history.json  : {job_name: [ {duration_seconds, duration_human,
    job_id, node, gres, finished_around} , ... ]} -- append-only, one entry
    added each time a job with that name is observed to have disappeared.
  - watch.log              : human-readable log of each poll and each
    detected finish, appended to (not rotated).

To use the estimates later: load duration_history.json, look at the samples
for a job name (or a name you expect to be similar/matching), and use
mean/median/max of `duration_seconds` as a rough estimate of how long until
that job (and therefore its GPU) is likely to finish.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


def parse_slurm_elapsed(elapsed: str) -> Optional[int]:
    """Parse SLURM's %M elapsed-time format into seconds.

    Handles all forms squeue emits: "MM:SS", "HH:MM:SS", "D-HH:MM:SS".
    Returns None if the string can't be parsed (e.g. empty/"INVALID").
    """
    elapsed = elapsed.strip()
    if not elapsed or elapsed.upper() in ("INVALID", "N/A", "UNLIMITED"):
        return None
    days = 0
    rest = elapsed
    if "-" in elapsed:
        day_part, rest = elapsed.split("-", 1)
        try:
            days = int(day_part)
        except ValueError:
            return None
    parts = rest.split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    elif len(parts) == 1:
        h, m, s = 0, 0, parts[0]
    else:
        return None
    return days * 86400 + h * 3600 + m * 60 + s


def human_duration(seconds: int) -> str:
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}-{h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_running_jobs() -> Dict[str, Dict[str, object]]:
    """Returns {job_id: {name, elapsed_seconds, elapsed_human, node, gres}} for
    every RUNNING job cluster-wide (all users)."""
    out = subprocess.run(
        ["squeue", "-h", "-t", "RUNNING", "-o", "%i|%j|%M|%N|%b"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    jobs: Dict[str, Dict[str, object]] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != 5:
            continue
        job_id, name, elapsed, node, gres = parts
        # Array jobs show as e.g. "12345_7"; keep as-is, it's still a unique id.
        seconds = parse_slurm_elapsed(elapsed)
        if seconds is None:
            continue
        jobs[job_id] = {
            "name": name,
            "elapsed_seconds": seconds,
            "elapsed_human": elapsed,
            "node": node,
            "gres": gres,
        }
    return jobs


def load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


MAX_LOG_BYTES = 100 * 1024 * 1024  # 100MB
TRIM_LOG_BYTES = 10 * 1024 * 1024  # 10MB


def trim_log_if_needed(log_path: Path, max_bytes: int = MAX_LOG_BYTES, trim_bytes: int = TRIM_LOG_BYTES) -> None:
    """If log_path exceeds max_bytes, drop the oldest ~trim_bytes from the front
    (rounded to the next newline so no line is left half-written)."""
    if not log_path.exists():
        return
    size = log_path.stat().st_size
    if size <= max_bytes:
        return
    with log_path.open("rb") as f:
        f.seek(trim_bytes)
        rest = f.read()
    nl = rest.find(b"\n")
    if nl != -1:
        rest = rest[nl + 1 :]
    with log_path.open("wb") as f:
        f.write(rest)


def log_line(log_path: Path, msg: str) -> None:
    ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    trim_log_if_needed(log_path)


def poll_once(state_path: Path, history_path: Path, log_path: Path) -> None:
    prev_state: Dict[str, Dict[str, object]] = load_json(state_path, {})
    history: Dict[str, list] = load_json(history_path, {})

    current = get_running_jobs()
    now_iso = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    finished_ids = [jid for jid in prev_state if jid not in current]
    for jid in finished_ids:
        info = prev_state[jid]
        name = str(info["name"])
        duration_seconds = int(info["elapsed_seconds"])
        history.setdefault(name, []).append(
            {
                "duration_seconds": duration_seconds,
                "duration_human": human_duration(duration_seconds),
                "job_id": jid,
                "node": info.get("node"),
                "gres": info.get("gres"),
                "finished_around": now_iso,
            }
        )
        log_line(
            log_path,
            f"FINISHED job_id={jid} name={name!r} last_elapsed={human_duration(duration_seconds)} "
            f"node={info.get('node')} gres={info.get('gres')}",
        )

    new_ids = [jid for jid in current if jid not in prev_state]
    for jid in new_ids:
        info = current[jid]
        log_line(
            log_path,
            f"NEW job_id={jid} name={info['name']!r} elapsed={info['elapsed_human']} "
            f"node={info['node']} gres={info['gres']}",
        )

    save_json(state_path, current)
    if finished_ids:
        save_json(history_path, history)
    log_line(log_path, f"poll done: {len(current)} running, {len(finished_ids)} finished, {len(new_ids)} new")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interval", type=int, default=3600, help="Seconds between polls (default 3600 = 1 hour)")
    p.add_argument(
        "--state-dir",
        default=str(Path(__file__).parent / "squeue_job_duration_watch"),
        help="Directory to store state/history/log files",
    )
    p.add_argument("--once", action="store_true", help="Run a single poll and exit (for testing/cron)")
    args = p.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "current_state.json"
    history_path = state_dir / "duration_history.json"
    log_path = state_dir / "watch.log"

    log_line(log_path, f"watch_squeue_job_durations starting, interval={args.interval}s, state_dir={state_dir}")

    if args.once:
        poll_once(state_path, history_path, log_path)
        return

    try:
        while True:
            try:
                poll_once(state_path, history_path, log_path)
            except subprocess.CalledProcessError as e:
                log_line(log_path, f"squeue call failed: {e}")
            except Exception as e:
                log_line(log_path, f"unexpected error during poll: {e!r}")
            time.sleep(int(args.interval))
    except KeyboardInterrupt:
        log_line(log_path, "stopped by KeyboardInterrupt")
        sys.exit(0)


if __name__ == "__main__":
    main()
