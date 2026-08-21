"""Strict provenance gate for Chapter-5 HW-HIL figures.

The gate deliberately has no model fallback. A figure may be labelled HW-HIL
only when all raw tables and their capture manifest are present, non-empty, and
internally consistent.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES: dict[str, tuple[str, ...]] = {
    "figure_5_1_waveforms.csv": (
        "capture_id", "evidence_level", "method", "board_id", "firmware_sha",
        "witness_mechanism", "timestamp_source", "sample_rate_hz", "time_us",
        "lease_delivery", "replay_trigger", "drive_gpio", "contact_current",
        "effect_seen", "obspub", "published_key", "reserve_commit",
        "terminal_commit", "reset_n", "force_safe_state", "closed_no_effect",
        "closed_with_effect", "commit", "w_c", "w_rec", "f_phys",
        "l_acc_marker", "l_eff_marker", "l_term_marker",
    ),
    "figure_5_2_reset_episodes.csv": (
        "episode_id", "evidence_level", "board_id", "firmware_sha", "reset_type",
        "timestamp_source", "reset_timestamp_us", "l_acc_timestamp_us",
        "drive_timestamp_us", "l_eff_timestamp_us", "l_term_timestamp_us",
        "probe_effect", "recovered_e_seen", "terminal_kind",
        "terminal_existed_pre_reset", "classification_ambiguous",
        "timing_uncertainty_us",
    ),
    "figure_5_3_latency_episodes.csv": (
        "matched_pair_id", "evidence_level", "board_id", "seed_cluster",
        "firmware_sha", "method", "timestamp_source", "request_timestamp_ns",
        "l_acc_timestamp_ns", "l_eff_timestamp_ns", "l_term_timestamp_ns",
        "reboot_entry_timestamp_ns", "recovery_close_timestamp_ns",
        "verification_ns", "durable_reserve_ns", "drive_to_stable_ns",
        "durable_finish_ns",
    ),
}

REQUIRED_MANIFEST_FIELDS = (
    "evidence_level", "capture_session_id", "captured_at_utc", "operator",
    "logic_analyzer_model", "logic_analyzer_serial", "clock_calibration",
    "firmware_sha", "board_serials", "probe_map", "raw_file_sha256",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def audit_hil_inputs(input_dir: Path) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    files: dict[str, Any] = {}
    loaded_rows: dict[str, list[dict[str, str]]] = {}

    manifest_path = input_dir / "capture_manifest.json"
    manifest: dict[str, Any] = {}
    if not manifest_path.exists():
        errors.append("missing capture_manifest.json")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid capture_manifest.json: {exc}")
        for field in REQUIRED_MANIFEST_FIELDS:
            if not manifest.get(field):
                errors.append(f"capture_manifest.json missing non-empty field: {field}")
        if manifest.get("evidence_level") != "HW-HIL":
            errors.append("capture_manifest.json evidence_level must be exactly HW-HIL")

    for filename, required_columns in REQUIRED_FILES.items():
        path = input_dir / filename
        status: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        files[filename] = status
        if not path.exists():
            errors.append(f"missing {filename}")
            continue
        try:
            columns, rows = _read_csv(path)
        except OSError as exc:
            errors.append(f"cannot read {filename}: {exc}")
            continue
        loaded_rows[filename] = rows
        missing_columns = [column for column in required_columns if column not in columns]
        status.update({
            "rows": len(rows),
            "columns": columns,
            "missing_columns": missing_columns,
            "sha256": _sha256(path),
        })
        if missing_columns:
            errors.append(f"{filename} missing columns: {', '.join(missing_columns)}")
        if not rows:
            errors.append(f"{filename} contains no observations")
        if any(row.get("evidence_level") != "HW-HIL" for row in rows):
            errors.append(f"{filename} has rows not labelled HW-HIL")
        expected_hash = (manifest.get("raw_file_sha256") or {}).get(filename)
        if manifest and not expected_hash:
            errors.append(f"capture_manifest.json lacks SHA-256 for {filename}")
        elif expected_hash and expected_hash.lower() != status["sha256"].lower():
            errors.append(f"SHA-256 mismatch for {filename}")

    waveform_rows = loaded_rows.get("figure_5_1_waveforms.csv", [])
    if waveform_rows:
        methods = {row.get("method") for row in waveform_rows}
        required_methods = {"SC-Post-HWM", "SAFE-Fuse"}
        if not required_methods.issubset(methods):
            errors.append("Figure 5.1 requires SC-Post-HWM and SAFE-Fuse captures")
        mechanisms = {
            row.get("witness_mechanism") for row in waveform_rows
            if row.get("method") in required_methods
        }
        if len(mechanisms) != 1 or not next(iter(mechanisms), ""):
            errors.append("Figure 5.1 paired captures must use one identical witness_mechanism")
        capture_ids = {row.get("capture_id") for row in waveform_rows}
        if len(capture_ids) < 2:
            warnings.append("Figure 5.1 should normally include one capture_id per paired method")

    reset_rows = loaded_rows.get("figure_5_2_reset_episodes.csv", [])
    if reset_rows:
        terminals = {row.get("terminal_kind") for row in reset_rows}
        allowed = {"Commit", "ClosedNoEffect", "ClosedWithEffect"}
        unexpected = terminals - allowed
        if unexpected:
            errors.append("Figure 5.2 unexpected terminal_kind values: " + ", ".join(sorted(unexpected)))
        if not {"0", "1"}.issupset({row.get("classification_ambiguous") for row in reset_rows}):
            errors.append("Figure 5.2 classification_ambiguous must be 0 or 1")

    latency_rows = loaded_rows.get("figure_5_3_latency_episodes.csv", [])
    if latency_rows:
        methods = {row.get("method") for row in latency_rows}
        if not {"SC-Post-HWM", "SAFE-Fuse"}.issubset(methods):
            errors.append("Figure 5.3 requires matched SC-Post-HWM and SAFE-Fuse episodes")
        pair_methods: dict[str, set[str]] = {}
        for row in latency_rows:
            pair_methods.setdefault(row.get("matched_pair_id", ""), set()).add(row.get("method", ""))
        incomplete = [pair_id for pair_id, values in pair_methods.items()
                      if values != {"SC-Post-HWM", "SAFE-Fuse"}]
        if incomplete:
            errors.append(f"Figure 5.3 has {len(incomplete)} incomplete matched pairs")

    return {
        "ready_for_hw_hil_figures": not errors,
        "input_dir": str(input_dir),
        "manifest": str(manifest_path),
        "files": files,
        "errors": errors,
        "warnings": warnings,
        "policy": "No MODEL/TIMING-MODEL fallback may be relabelled HW-HIL.",
    }

