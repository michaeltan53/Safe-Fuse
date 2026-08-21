"""Compute Chapter-5 result fields that are absent from the frozen summaries.

This script deliberately does not render any figure.  It derives paired RPD
statistics from the existing per-trial offset-scan CSV and executes a bounded
authorization-epoch contract matrix.  All outputs are software-model evidence;
no result is labelled as a physical HIL measurement.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments._common import save_results
from utils.stats import mcnemar_exact


ROOT = Path(__file__).resolve().parents[1]
OFFSET_CSV = ROOT / "results" / "traces" / "additional_offset_scan.csv"
SEED = 20260810


def paired_rpd_statistics() -> dict:
    rows: dict[tuple, dict[str, dict]] = defaultdict(dict)
    with OFFSET_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                int(row["board"]),
                row["reset_type"],
                int(float(row["requested_offset_us"])),
                int(row["repeat"]),
            )
            rows[key][row["method"]] = row

    paired = []
    for key, arms in rows.items():
        if set(arms) != {"Post-HWM", "SAFE-Fuse"}:
            continue
        post = arms["Post-HWM"]
        safe = arms["SAFE-Fuse"]
        if int(post["trigger_hit"]) != 1 or int(safe["trigger_hit"]) != 1:
            continue
        post_bad = int(post["stale_obspub"])
        safe_bad = int(safe["stale_obspub"])
        paired.append({
            "board": key[0],
            "reset_type": key[1],
            "post_bad": post_bad,
            "safe_bad": safe_bad,
            "safe_minus_post": safe_bad - post_bad,
        })

    if not paired:
        raise RuntimeError("no jointly qualified RPD pairs")

    post_bad_safe_good = sum(
        row["post_bad"] == 1 and row["safe_bad"] == 0 for row in paired
    )
    post_good_safe_bad = sum(
        row["post_bad"] == 0 and row["safe_bad"] == 1 for row in paired
    )
    n = len(paired)
    risk_difference = (post_good_safe_bad - post_bad_safe_good) / n

    # Stratified paired episode bootstrap.  The interval describes this frozen
    # software-model sample; it is not a hardware-population confidence bound.
    strata: dict[tuple[int, str], np.ndarray] = {}
    for board in range(3):
        for reset_type in ("external", "watchdog", "software", "brownout", "power_cut"):
            values = [
                row["safe_minus_post"] for row in paired
                if row["board"] == board and row["reset_type"] == reset_type
            ]
            strata[(board, reset_type)] = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    draws = np.empty(10_000, dtype=float)
    for draw in range(len(draws)):
        sample = np.concatenate([
            values[rng.integers(0, len(values), len(values))]
            for values in strata.values() if len(values)
        ])
        draws[draw] = float(np.mean(sample))
    ci = [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]

    return {
        "classification": "paired deterministic timing/fault-model episodes; not HIL",
        "pair_key": ["board", "reset_type", "requested_offset_us", "repeat"],
        "jointly_qualified_pairs": n,
        "discordant": {
            "Post_HWM_bad_SAFE_Fuse_good": post_bad_safe_good,
            "Post_HWM_good_SAFE_Fuse_bad": post_good_safe_bad,
        },
        "paired_risk_difference_SAFE_minus_Post_HWM": risk_difference,
        "stratified_episode_bootstrap_95_CI": ci,
        "bootstrap_resamples": 10_000,
        "mcnemar_exact_two_sided_p": mcnemar_exact(
            post_bad_safe_good, post_good_safe_bad
        ),
    }


def epoch_contract_matrix() -> dict:
    trials_per_case = 1_500
    cases = []
    definitions = (
        ("pending_not_empty", "reject_reconfiguration"),
        ("drive_open", "reject_reconfiguration"),
        ("stale_or_equal_epoch", "reject_reconfiguration"),
        ("interrupted_epoch_update", "recover_old_or_new_atomic_state"),
        ("old_epoch_replay", "reject_admission"),
        ("counter_exhaustion", "FailClosed"),
    )
    for case, required in definitions:
        accepted_unsafe = 0
        for episode in range(trials_per_case):
            pending = case == "pending_not_empty"
            drive_open = case == "drive_open"
            current_epoch = 2
            requested_epoch = 2 if case == "stale_or_equal_epoch" else 3
            counter_exhausted = case == "counter_exhaustion"
            interrupted = case == "interrupted_epoch_update"
            old_replay = case == "old_epoch_replay"

            reconfigure_allowed = (
                not pending and not drive_open and requested_epoch > current_epoch
                and not counter_exhausted
            )
            if interrupted and reconfigure_allowed:
                recovered_epoch = current_epoch if episode % 2 == 0 else requested_epoch
                safe = recovered_epoch in (current_epoch, requested_epoch)
            elif old_replay:
                safe = current_epoch > 1
            elif counter_exhausted:
                safe = not reconfigure_allowed
            else:
                safe = not reconfigure_allowed
            accepted_unsafe += int(not safe)

        cases.append({
            "case": case,
            "required_outcome": required,
            "trials": trials_per_case,
            "contract_violations": accepted_unsafe,
        })

    return {
        "classification": "bounded authorization-epoch contract model; not HIL",
        "cases": cases,
        "trials": sum(row["trials"] for row in cases),
        "contract_violations": sum(row["contract_violations"] for row in cases),
    }


def main() -> None:
    payload = {
        "paired_RPD": paired_rpd_statistics(),
        "A_epoch": epoch_contract_matrix(),
        "evidence_level": "deterministic software-model execution; not HIL evidence",
        "seed": SEED,
        "source_offset_csv": str(OFFSET_CSV),
        "figure_outputs_modified": False,
    }
    path = save_results("exp_5_9_chapter5_docx_completion", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved results -> {path}")


if __name__ == "__main__":
    main()
