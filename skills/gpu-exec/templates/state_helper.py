#!/usr/bin/env python3
"""state_helper.py — monitor.sh JSON state/spec helper.

Replaces all inline python3 -c calls in monitor.sh so there are no
quoting/escaping issues with embedded paths or values.

All commands exit 0 on success, 1 on error (message to stderr).
"""
import sys
import json
import datetime


def load(path):
    with open(path) as f:
        return json.load(f)


def dump(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def now():
    return datetime.datetime.utcnow().isoformat() + "Z"


# ── State file commands ───────────────────────────────────────────────────────

def cmd_get_state(args):
    """get_state <state_file>"""
    print(load(args[0])["state"])


def cmd_set_state(args):
    """set_state <state_file> <new_state>"""
    path, new_state = args[0], args[1]
    s = load(path)
    s["state"] = new_state
    if not s.get("started_at"):
        s["started_at"] = now()
    s["updated_at"] = now()
    dump(s, path)


def cmd_get_field(args):
    """get_field <state_file> <key>"""
    print(load(args[0]).get(args[1], ""))


def cmd_set_field(args):
    """set_field <state_file> <key> [value]"""
    path, key = args[0], args[1]
    value = args[2] if len(args) > 2 else ""
    s = load(path)
    s[key] = value
    dump(s, path)


def cmd_increment_retry(args):
    """increment_retry <state_file>"""
    path = args[0]
    s = load(path)
    s["retry_count"] = s.get("retry_count", 0) + 1
    dump(s, path)


def cmd_record_phase_result(args):
    """record_phase_result <state_file> <phase_id>"""
    path, phase_id = args[0], args[1]
    s = load(path)
    s.setdefault("phase_results", {})[phase_id] = {
        "state": "PHASE_DONE",
        "gate": "PASS",
        "at": now(),
    }
    dump(s, path)


# ── Time command ──────────────────────────────────────────────────────────────

def cmd_elapsed_hours(args):
    """elapsed_hours <started_at_iso>"""
    started_at = args[0].rstrip("Z")
    try:
        start = datetime.datetime.fromisoformat(started_at)
        elapsed = (datetime.datetime.utcnow() - start).total_seconds() / 3600
        print(int(elapsed))
    except Exception:
        print(0)


# ── Spec file commands ────────────────────────────────────────────────────────

def cmd_get_mode(args):
    """get_mode <spec_file>"""
    print(load(args[0]).get("mode", "runpod"))


def cmd_phase_count(args):
    """phase_count <spec_file>"""
    print(len(load(args[0]).get("phases", [])))


def cmd_get_phase(args):
    """get_phase <spec_file> <idx> <field>"""
    spec_path, idx_str, field = args[0], args[1], args[2]
    spec = load(spec_path)
    idx = int(idx_str)
    phases = spec.get("phases", [])
    if idx < len(phases):
        print(phases[idx].get(field, ""))


def cmd_get_upload_dest(args):
    """get_upload_dest <spec_file> <idx>"""
    spec_path, idx_str = args[0], args[1]
    spec = load(spec_path)
    idx = int(idx_str)
    phases = spec.get("phases", [])
    dest = "/workspace/project"
    if idx < len(phases):
        dest = phases[idx].get("upload", {}).get("remote_dest", dest)
    print(dest)


def cmd_get_upload_paths(args):
    """get_upload_paths <spec_file> <idx>  — prints one path per line (no prefix)"""
    spec_path, idx_str = args[0], args[1]
    spec = load(spec_path)
    idx = int(idx_str)
    phases = spec.get("phases", [])
    if idx < len(phases):
        for lp in phases[idx].get("upload", {}).get("local_paths", []):
            print(lp.rstrip("/"))


def cmd_get_gate_meta(args):
    """get_gate_meta <spec_file> <idx>  — prints metrics_file on line 1, condition on line 2"""
    spec_path, idx_str = args[0], args[1]
    spec = load(spec_path)
    idx = int(idx_str)
    phases = spec.get("phases", [])
    gate = phases[idx].get("gate", {}) if idx < len(phases) else {}
    print(gate.get("metrics_file", ""))
    print(gate.get("condition", ""))


def cmd_get_outputs(args):
    """get_outputs <spec_file> <idx>  — prints one output path per line"""
    spec_path, idx_str = args[0], args[1]
    spec = load(spec_path)
    idx = int(idx_str)
    phases = spec.get("phases", [])
    if idx < len(phases):
        for o in phases[idx].get("outputs", []):
            print(o)


# ── Dispatch ──────────────────────────────────────────────────────────────────

COMMANDS = {
    "get_state":           cmd_get_state,
    "set_state":           cmd_set_state,
    "get_field":           cmd_get_field,
    "set_field":           cmd_set_field,
    "increment_retry":     cmd_increment_retry,
    "record_phase_result": cmd_record_phase_result,
    "elapsed_hours":       cmd_elapsed_hours,
    "get_mode":            cmd_get_mode,
    "phase_count":         cmd_phase_count,
    "get_phase":           cmd_get_phase,
    "get_upload_dest":     cmd_get_upload_dest,
    "get_upload_paths":    cmd_get_upload_paths,
    "get_gate_meta":       cmd_get_gate_meta,
    "get_outputs":         cmd_get_outputs,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        cmds = ", ".join(sorted(COMMANDS))
        print(
            f"Usage: state_helper.py <command> [args...]\nCommands: {cmds}",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        COMMANDS[sys.argv[1]](sys.argv[2:])
    except Exception as e:
        print(f"ERROR [{sys.argv[1]}]: {e}", file=sys.stderr)
        sys.exit(1)
