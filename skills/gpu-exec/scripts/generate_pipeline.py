#!/usr/bin/env python3
"""
generate_pipeline.py — /gpu-exec skill pipeline generator

Usage:
  python3 generate_pipeline.py --spec .omc/gpu-exec/pipeline_spec.json [--project-dir .]
  python3 generate_pipeline.py --spec ... --status
  python3 generate_pipeline.py --spec ... --resume
"""
import argparse
import datetime
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


SKILL_DIR = Path(__file__).parent.parent
TEMPLATES_DIR = SKILL_DIR / "templates"


# ── Template substitution ──────────────────────────────────────────────────────

def generate_from_template(tmpl_path: Path, subs: dict) -> str:
    """Substitute __KEY__ tokens; bash $VAR, $(cmd), ${VAR} are untouched."""
    text = tmpl_path.read_text()
    for key, val in subs.items():
        text = text.replace(f"__{key}__", str(val))
    return text


# ── Crontab management ─────────────────────────────────────────────────────────

def register_crontab(monitor_sh: Path, log_file: Path, pipeline_id: str) -> bool:
    """Idempotent crontab registration — removes old entry before adding new."""
    try:
        existing = subprocess.check_output(
            ["crontab", "-l"], stderr=subprocess.DEVNULL
        ).decode()
    except subprocess.CalledProcessError:
        existing = ""

    marker = f"# gpu-exec-{pipeline_id}"
    lines = [l for l in existing.splitlines() if marker not in l]
    new_entry = f"*/10 * * * * bash {monitor_sh} >> {log_file} 2>&1 {marker}"
    lines.append(new_entry)
    proc = subprocess.run(
        ["crontab", "-"],
        input="\n".join(lines) + "\n",
        text=True,
    )
    return proc.returncode == 0


def remove_crontab(pipeline_id: str) -> None:
    marker = f"# gpu-exec-{pipeline_id}"
    try:
        existing = subprocess.check_output(
            ["crontab", "-l"], stderr=subprocess.DEVNULL
        ).decode()
    except subprocess.CalledProcessError:
        return
    lines = [l for l in existing.splitlines() if marker not in l]
    subprocess.run(["crontab", "-"], input="\n".join(lines) + "\n", text=True)


# ── Status ────────────────────────────────────────────────────────────────────

def cmd_status(omc_dir: Path, pipeline_id: str) -> None:
    state_file = omc_dir / "state.json"
    if not state_file.exists():
        print("No state.json found — pipeline not generated yet.")
        sys.exit(1)

    # Read state.json with shared flock (timeout 5s)
    import fcntl
    with open(state_file) as f:
        try:
            fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            # Flock unavailable — read without lock (best effort)
            pass
        state = json.load(f)

    print(f"\n=== gpu-exec status: {pipeline_id} ===")
    print(f"  State:   {state.get('state', '?')}")
    print(f"  Mode:    {state.get('mode', '?')}")
    print(f"  Phase:   {state.get('current_phase_index', 0)}")
    print(f"  Pod ID:  {state.get('pod_id') or 'none'}")
    print(f"  Retries: {state.get('retry_count', 0)}")
    print(f"  Updated: {state.get('updated_at', '?')}")

    # Check crontab
    marker = f"# gpu-exec-{pipeline_id}"
    try:
        crontab = subprocess.check_output(["crontab", "-l"], stderr=subprocess.DEVNULL).decode()
        registered = marker in crontab
    except subprocess.CalledProcessError:
        registered = False
    print(f"  Cron:    {'registered' if registered else 'NOT registered'}")
    print()


# ── Resume ────────────────────────────────────────────────────────────────────

def cmd_resume(omc_dir: Path, pipeline_id: str, monitor_sh: Path, log_file: Path) -> None:
    state_file = omc_dir / "state.json"
    if not state_file.exists():
        print("No state.json found.")
        sys.exit(1)

    with open(state_file) as f:
        state = json.load(f)

    current_state = state.get("state", "")
    if current_state not in ("GATE_FAIL", "ERROR"):
        print(f"Resume only valid from GATE_FAIL or ERROR state (current: {current_state})")
        sys.exit(1)

    if current_state == "GATE_FAIL":
        # Keep current phase, reset to IDLE
        pod_id = state.get("pod_id")
        if pod_id:
            print(f"WARNING: A pod ({pod_id}) may still be running. Delete it manually if needed.")
        state["state"] = "IDLE"
        state["retry_count"] = 0
        state["updated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
        print(f"Resuming from phase {state['current_phase_index']} (GATE_FAIL → IDLE)")
    else:
        # ERROR: full restart
        pod_id = state.get("pod_id")
        if pod_id:
            print(f"WARNING: A pod ({pod_id}) may still be running. Delete it manually if needed.")
        state["state"] = "IDLE"
        state["current_phase_index"] = 0
        state["retry_count"] = 0
        state["pod_id"] = None
        state["pod_ip"] = None
        state["updated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
        print("Full restart from phase 0 (ERROR → IDLE)")

    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

    # Re-register crontab
    if register_crontab(monitor_sh, log_file, pipeline_id):
        print(f"Cron re-registered for {pipeline_id}")
    else:
        print("WARNING: crontab registration failed")


# ── Generate ──────────────────────────────────────────────────────────────────

def cmd_generate(spec_path: Path, project_dir: Path) -> None:
    spec = json.loads(spec_path.read_text())
    pipeline_id = spec["pipeline_id"]
    mode = spec.get("mode", "runpod")

    omc_dir = project_dir / ".omc" / "gpu-exec" / pipeline_id
    omc_dir.mkdir(parents=True, exist_ok=True)

    monitor_sh = omc_dir / "monitor.sh"
    gate_eval_py = omc_dir / "gate_eval.py"
    state_file = omc_dir / "state.json"
    log_file = omc_dir / "monitor.log"
    env_file = omc_dir / ".env"

    # Build substitution dict
    runpod = spec.get("runpod", {})
    subs = {
        "PIPELINE_ID":     pipeline_id,
        "PROJECT_DIR":     str(project_dir.resolve()),
        "STATE_FILE":      str(state_file),
        "LOG_FILE":        str(log_file),
        "NOTIFY_EMAIL":    spec.get("notifications", {}).get("email", ""),
        "GPU_TYPE_ID":     runpod.get("gpu_type_id", ""),
        "POD_IMAGE":       runpod.get("image_name", ""),
        "NETWORK_VOL_ID":  runpod.get("network_volume_id", ""),
        "SSH_KEY_PATH":    os.environ.get("SSH_KEY_PATH", os.path.expanduser("~/.ssh/id_ed25519")),
        "SPEC_FILE":       str(spec_path.resolve()),
        "ENV_FILE":        str(env_file),
        "MAX_RUNTIME_HOURS": str(spec.get("max_runtime_hours", "")),
    }

    # Local mode: blank out RunPod fields
    if mode == "local":
        subs.update({
            "GPU_TYPE_ID":    "",
            "POD_IMAGE":      "",
            "NETWORK_VOL_ID": "",
            "SSH_KEY_PATH":   "",
        })

    # Generate monitor.sh
    monitor_tmpl = TEMPLATES_DIR / "monitor.sh.tmpl"
    monitor_content = generate_from_template(monitor_tmpl, subs)
    monitor_sh.write_text(monitor_content)
    monitor_sh.chmod(monitor_sh.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Generated: {monitor_sh}")

    # Copy gate_eval.py (no __KEY__ substitution — copied as-is)
    gate_src = TEMPLATES_DIR / "gate_eval.py"
    shutil.copy(gate_src, gate_eval_py)
    gate_eval_py.chmod(gate_eval_py.stat().st_mode | stat.S_IEXEC)
    print(f"Generated: {gate_eval_py}")

    # Copy state_helper.py (no __KEY__ substitution — called by monitor.sh)
    state_helper_src = TEMPLATES_DIR / "state_helper.py"
    state_helper_dst = omc_dir / "state_helper.py"
    shutil.copy(state_helper_src, state_helper_dst)
    state_helper_dst.chmod(state_helper_dst.stat().st_mode | stat.S_IEXEC)
    print(f"Generated: {state_helper_dst}")

    # Create .env file (chmod 600)
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key and mode == "runpod":
        print("WARNING: RUNPOD_API_KEY not set in environment — .env will be empty")
    env_file.write_text(f"RUNPOD_API_KEY={api_key}\n")
    env_file.chmod(0o600)
    print(f"Generated: {env_file} (chmod 600)")

    # Initialize state.json (only if not exists or IDLE)
    if not state_file.exists():
        state = {
            "pipeline_id": pipeline_id,
            "state": "IDLE",
            "mode": mode,
            "current_phase_index": 0,
            "pod_id": None,
            "pod_ip": None,
            "phase_results": {},
            "retry_count": 0,
            "started_at": None,
            "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
        }
        state_file.write_text(json.dumps(state, indent=2))
        print(f"Initialized: {state_file}")
    else:
        print(f"Kept existing: {state_file} (delete manually to reset)")

    # Register crontab
    if register_crontab(monitor_sh, log_file, pipeline_id):
        print(f"Crontab registered: */10 * * * * bash {monitor_sh}  # gpu-exec-{pipeline_id}")
    else:
        print("WARNING: crontab registration failed — register manually")
        print(f"  */10 * * * * bash {monitor_sh} >> {log_file} 2>&1 # gpu-exec-{pipeline_id}")

    print(f"\nPipeline '{pipeline_id}' ready. Monitor will run every 10 minutes.")
    print(f"Check status:  python3 {__file__} --spec {spec_path} --status")
    print(f"After GATE_FAIL: python3 {__file__} --spec {spec_path} --resume")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="gpu-exec pipeline generator")
    parser.add_argument("--spec", required=True, help="Path to pipeline_spec.json")
    parser.add_argument("--project-dir", default=".", help="Project root directory")
    parser.add_argument("--status", action="store_true", help="Show pipeline status")
    parser.add_argument("--resume", action="store_true", help="Resume after GATE_FAIL or ERROR")
    args = parser.parse_args()

    spec_path = Path(args.spec).resolve()
    project_dir = Path(args.project_dir).resolve()

    if not spec_path.exists():
        print(f"ERROR: spec file not found: {spec_path}")
        sys.exit(1)

    spec = json.loads(spec_path.read_text())
    pipeline_id = spec["pipeline_id"]
    omc_dir = project_dir / ".omc" / "gpu-exec" / pipeline_id
    monitor_sh = omc_dir / "monitor.sh"
    log_file = omc_dir / "monitor.log"

    if args.status:
        cmd_status(omc_dir, pipeline_id)
    elif args.resume:
        cmd_resume(omc_dir, pipeline_id, monitor_sh, log_file)
    else:
        cmd_generate(spec_path, project_dir)


if __name__ == "__main__":
    main()
