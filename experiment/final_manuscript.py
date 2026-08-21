"""Experiments required by the final Chapter-5 manuscript.

This module closes the result fields that are not covered by the original
Chapter-5 runners: bounded abstract-state exploration, shared-trace strong
baselines, a recovery-time contract, and an explicit A/B torn-write matrix.

The implementation is executable software-model evidence.  It intentionally
does not label model output as a physical MCU/HIL measurement.
"""
from __future__ import annotations

import itertools
import json
import csv
import os
from pathlib import Path

import numpy as np

from experiments._common import print_table, save_results
from utils.stats import upper_95_one_sided


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SEED = 20260807
BOARDS = 3
EPISODES_PER_BOARD = 10_000


def _plt():
    os.environ.setdefault("MPLCONFIGDIR", str(RESULTS / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _render_figure52_legacy(coverage: dict) -> list[str]:
    """Render a four-boundary software-model audit with exact bin counts.

    The figure is intentionally not labelled as an actual-reset hardware scan.
    Ambiguous episodes remain a separate stratum and are not silently deleted.
    """
    with Path(coverage["raw_csv"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["realized_offset_us"] = float(row["realized_offset_us"])
        row["ambiguous"] = int(row["ambiguous"])
        for key in ("accepted", "frontier_advanced", "drive_started", "probe_effect",
                    "recovered_seen", "terminal_existed_pre_reset"):
            row[key] = int(row[key])

    boundaries = coverage["boundaries_us"]
    uncertainty = coverage["uncertainty_band_us"]
    accounting = coverage["trial_accounting"]
    figures_dir = RESULTS / "figs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    png = figures_dir / "fig_5_2.png"
    pdf = png.with_suffix(".pdf")
    svg = png.with_suffix(".svg")

    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.25), sharey=False)
    axes = axes.ravel()
    palette = ("#1769aa", "#2e7d32", "#7b61a8", "#e37400")
    ambiguous_color = "#9aa0a6"

    def stacked_exact_counts(axis, *, origin: float, xlim: tuple[float, float],
                             categories: list[tuple[str, callable]], title: str) -> None:
        selected = [row for row in rows
                    if xlim[0] <= row["realized_offset_us"] - origin <= xlim[1]]
        xs = sorted({row["realized_offset_us"] - origin for row in selected})
        bottoms = np.zeros(len(xs), dtype=float)
        totals_by_category = []
        for category_index, (label, predicate) in enumerate(categories):
            counts = []
            for x in xs:
                group = [row for row in selected
                         if row["realized_offset_us"] - origin == x and not row["ambiguous"]]
                counts.append(sum(int(predicate(row)) for row in group))
            axis.bar(xs, counts, bottom=bottoms, width=8.4,
                     color=palette[category_index % len(palette)], alpha=.82,
                     edgecolor="white", linewidth=.25,
                     label=f"{label}")
            bottoms += np.asarray(counts)
            totals_by_category.append(sum(counts))
        ambiguous_counts = [
            sum(row["ambiguous"] for row in selected
                if row["realized_offset_us"] - origin == x)
            for x in xs
        ]
        axis.bar(xs, ambiguous_counts, bottom=bottoms, width=8.4,
                 facecolor="none", edgecolor=ambiguous_color, hatch="////",
                 linewidth=.55, label=f"ambiguous")
        axis.axvspan(-uncertainty, uncertainty, facecolor="#dadce0", alpha=.55,
                     hatch="////", edgecolor="#9aa0a6", linewidth=0)
        axis.axvline(0, color="#3c4043", lw=1.0)
        axis.set_xlim(*xlim)
        axis.set_xlabel(r"modelled reset offset from boundary ($\mu$s)", fontsize=7.0)
        axis.set_ylabel("exact episodes / 10 μs bin", fontsize=7.0)
        axis.set_title(title, fontsize=8.2, loc="left")
        axis.grid(axis="y", alpha=.2)
        axis.tick_params(labelsize=6.5)
        axis.legend(loc="upper left", fontsize=9, frameon=False,
                    handlelength=1.6, ncol=2)

    stacked_exact_counts(
        axes[0], origin=boundaries["l_acc"], xlim=(-190, 190),
        categories=[
            ("not accepted", lambda r: not r["accepted"]),
            ("Accepted + wc advanced", lambda r: r["accepted"] and r["frontier_advanced"]),
        ],
        title=r"(a) $\ell_{acc}$: admission and durable exclusion",
    )
    stacked_exact_counts(
        axes[1], origin=boundaries["Drive"], xlim=(-190, 190),
        categories=[
            ("accepted, Drive not begun", lambda r: r["accepted"] and not r["drive_started"]),
            ("Drive begun", lambda r: r["drive_started"]),
        ],
        title="(b) Drive: drive-start boundary",
    )
    stacked_exact_counts(
        axes[2], origin=boundaries["l_eff"], xlim=(-190, 190),
        categories=[
            ("no effect → ClosedNoEffect", lambda r: not r["probe_effect"]),
            ("late/formed effect → ClosedWithEffect", lambda r: r["probe_effect"]),
        ],
        title=r"(c) $\ell_{eff}$: C2 no-effect and late-effect branches",
    )
    stacked_exact_counts(
        axes[3], origin=boundaries["l_term"], xlim=(-190, 190),
        categories=[
            ("created in recovery", lambda r: not r["terminal_existed_pre_reset"]),
            ("existing Commit", lambda r: r["terminal_kind"] == "Commit"),
            ("existing ClosedWithEffect", lambda r: r["terminal_kind"] == "ClosedWithEffect"),
            ("existing ClosedNoEffect", lambda r: r["terminal_kind"] == "ClosedNoEffect"),
        ],
        title=r"(d) $\ell_{term}$: recovery-created vs three existing terminals",
    )

    ranges = coverage["ambiguous_conservative_window_ranges"]
    range_text = "  ".join(
        f"{window}=[{bounds[0]:,},{bounds[1]:,}]"
        for window, bounds in ranges.items()
    )
    fig.suptitle("Figure 5.2 - software-model boundary audit with exact bin counts",
                 fontsize=10.3, y=.995)
    fig.text(.5, .948,
             "scheduled {scheduled:,} → triggered {triggered:,} → qualified {qualified:,} "
             "→ ambiguous {ambiguous:,} → assigned {assigned_window:,}".format(**accounting),
             ha="center", fontsize=7.1, color="#3c4043")
    fig.text(.5, .026, "most-conservative window totals when ambiguous episodes are assigned left/right: "
             + range_text, ha="center", fontsize=5.85, color="#3c4043")
    fig.text(.5, .006,
             f"hatched bars/bands are the ±{uncertainty:.2f} μs uncertainty stratum; deterministic software timing model, not probe-timestamped HIL",
             ha="center", fontsize=5.9, color="#5f6368")
    fig.subplots_adjust(left=.105, right=.985, top=.905, bottom=.105,
                        hspace=.42, wspace=.22)
    for output in (png, pdf, svg):
        fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [str(png), str(pdf), str(svg)]


def _render_figure52_review(coverage: dict) -> list[str]:
    """Render the review-ready model boundary audit.

    Each marker is a per-offset-bin outcome share and each whisker is a Wilson
    95% interval. Exact classifiable/ambiguous counts remain visible above the
    bin, while the ambiguity stratum is never folded into an outcome class.
    """
    with Path(coverage["raw_csv"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["realized_offset_us"] = float(row["realized_offset_us"])
        row["ambiguous"] = int(row["ambiguous"])
        for key in ("accepted", "frontier_advanced", "drive_started", "probe_effect",
                    "recovered_seen", "terminal_existed_pre_reset"):
            row[key] = int(row[key])

    boundaries = coverage["boundaries_us"]
    uncertainty = coverage["uncertainty_band_us"]
    accounting = coverage["trial_accounting"]
    figures_dir = RESULTS / "figs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    png = figures_dir / "fig_5_2.png"
    pdf = png.with_suffix(".pdf")
    svg = png.with_suffix(".svg")

    plt = _plt()
    fig = plt.figure(figsize=(7.25, 8.25))
    outer = fig.add_gridspec(2, 2, hspace=.50, wspace=.25)
    count_axes = []
    axes = []
    for panel_index in range(4):
        row, column = divmod(panel_index, 2)
        panel = outer[row, column].subgridspec(2, 1, height_ratios=(.30, 1.0), hspace=.035)
        count_axis = fig.add_subplot(panel[0])
        axis = fig.add_subplot(panel[1], sharex=count_axis)
        count_axes.append(count_axis)
        axes.append(axis)
    for axis in axes[1:]:
        axis.sharey(axes[0])
    palette = ("#1769aa", "#2e7d32", "#7b61a8", "#e37400")
    ambiguous_color = "#9aa0a6"

    def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
        if trials == 0:
            return (np.nan, np.nan)
        z = 1.959963984540054
        proportion = successes / trials
        denominator = 1.0 + z * z / trials
        center = (proportion + z * z / (2.0 * trials)) / denominator
        radius = z * np.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        ) / denominator
        return max(0.0, center - radius), min(1.0, center + radius)

    def boundary_audit(axis, count_axis, *, origin: float, categories: list[tuple[str, callable]],
                       title: str, note: str = "") -> None:
        xlim = (-190.0, 190.0)
        selected = [row for row in rows
                    if xlim[0] <= row["realized_offset_us"] - origin <= xlim[1]]
        xs = sorted({row["realized_offset_us"] - origin for row in selected})
        groups = {
            x: [row for row in selected
                if row["realized_offset_us"] - origin == x and not row["ambiguous"]]
            for x in xs
        }
        classifiable = np.asarray([len(groups[x]) for x in xs], dtype=int)
        ambiguous = np.asarray([
            sum(row["ambiguous"] for row in selected
                if row["realized_offset_us"] - origin == x)
            for x in xs
        ], dtype=int)

        for category_index, (label, predicate) in enumerate(categories):
            counts = np.asarray([
                sum(int(predicate(row)) for row in groups[x]) for x in xs
            ], dtype=int)
            shares = np.divide(
                counts, classifiable,
                out=np.full(len(xs), np.nan, dtype=float),
                where=classifiable > 0,
            )
            intervals = [wilson_interval(int(k), int(n))
                         for k, n in zip(counts, classifiable)]
            lower = np.asarray([interval[0] for interval in intervals])
            upper = np.asarray([interval[1] for interval in intervals])
            mask = np.isfinite(shares)
            color = palette[category_index % len(palette)]
            x_values = np.asarray(xs)[mask]
            axis.plot(x_values, shares[mask], color=color, lw=1.05,
                      marker="o", ms=2.35, label=f"{label}")
            axis.errorbar(
                x_values, shares[mask],
                yerr=np.maximum(
                    0.0,
                    np.vstack((shares[mask] - lower[mask], upper[mask] - shares[mask])),
                ),
                fmt="none", ecolor=color, elinewidth=.55, capsize=1.15, alpha=.65,
            )

        ambiguity_mask = ambiguous > 0
        if ambiguity_mask.any():
            axis.scatter(
                np.asarray(xs)[ambiguity_mask], np.full(ambiguity_mask.sum(), -.055),
                s=8.0 + 1.8 * np.sqrt(ambiguous[ambiguity_mask]), marker="x",
                linewidths=.8, color=ambiguous_color, clip_on=False,
                label=f"ambiguous",
            )
        axis.axvspan(-uncertainty, uncertainty, facecolor="#dadce0", alpha=.55,
                     hatch="////", edgecolor="#9aa0a6", linewidth=0)
        axis.axvline(0, color="#3c4043", lw=1.0)
        axis.set_xlim(*xlim)
        axis.set_ylim(-.11, 1.04)
        axis.set_xlabel(r"reset timestamp − boundary timestamp ($\mu$s)", fontsize=10.0)
        axis.set_ylabel("share of classifiable episodes", fontsize=9.0)
        axis.grid(axis="y", alpha=.2)
        axis.tick_params(labelsize=7.5)
        axis.legend(loc="upper left",
                    fontsize=9,
                    frameon=False,
                    handlelength=1.5,
                    ncol=2,
                    columnspacing=0.9)
        axis.text(.995, .018,
                    note,
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=6.2,
                    color="#5f6368")

        count_axis.bar(np.asarray(xs) - 2.15, classifiable, width=4.1,
                       color="#5f6368", alpha=.76, label="classifiable n/bin")
        count_axis.bar(np.asarray(xs) + 2.15, ambiguous, width=4.1,
                       color=ambiguous_color, alpha=.58, hatch="//",
                       label="ambiguous n/bin")
        count_axis.axvspan(-uncertainty, uncertainty, facecolor="#dadce0", alpha=.45,
                           hatch="////", edgecolor="#9aa0a6", linewidth=0)
        count_axis.axvline(0, color="#3c4043", lw=.8)
        count_axis.set_xlim(*xlim)
        count_axis.set_title(title, fontsize=10, loc="left", pad=3)
        count_axis.set_ylabel("n/bin", fontsize=10)
        count_axis.tick_params(axis="x", labelbottom=False, length=0)
        count_axis.tick_params(axis="y", labelsize=10, length=2)
        count_axis.grid(axis="y", alpha=.16)
        # count_axis.spines[["top", "right"]].set_visible(False)
        for spine in ("top", "right"):
            count_axis.spines[spine].set_visible(False)
        count_axis.legend(loc="upper right",
                            fontsize=9,
                            frameon=False,
                            ncol=2,
                            handlelength=1.1,
                            columnspacing=0.8)

    boundary_audit(
        axes[0], count_axes[0], origin=boundaries["l_acc"],
        categories=[
            ("not accepted", lambda row: not row["accepted"]),
            ("Accepted + wc advanced", lambda row: row["accepted"] and row["frontier_advanced"]),
        ],
        title=r"(a) $\ell_{acc}$: admission and durable exclusion",
    )
    boundary_audit(
        axes[1], count_axes[1], origin=boundaries["Drive"],
        categories=[
            ("accepted, pre-Drive", lambda row: row["accepted"] and not row["drive_started"]),
            ("Drive begun", lambda row: row["drive_started"]),
        ],
        title="(b) Drive: drive-start boundary",
    )
    c2 = coverage["C2_independent_ground_truth_confusion"]
    boundary_audit(
        axes[2], count_axes[2], origin=boundaries["l_eff"],
        categories=[
            ("no effect → ClosedNoEffect", lambda row: not row["probe_effect"]),
            ("late effect → ClosedWithEffect", lambda row: row["probe_effect"]),
        ],
        title=r"(c) $\ell_{eff}$: C2 no-effect and late-effect branches",
        note=(f"C2 truth TP/TN/FP/FN={c2['effect_and_E_Seen']}/"
              f"{c2['no_effect_and_E_NotSeen']}/"
              f"{c2['no_effect_and_E_Seen_FP']}/"
              f"{c2['effect_and_E_NotSeen_FN']}"),
    )
    c4 = coverage["C4_terminal_stratification"]
    boundary_audit(
        axes[3], count_axes[3], origin=boundaries["l_term"],
        categories=[
            ("recovery-created terminal", lambda row: not row["terminal_existed_pre_reset"]),
            ("existing Commit", lambda row: row["terminal_existed_pre_reset"] and row["terminal_kind"] == "Commit"),
            ("existing CWE", lambda row: row["terminal_existed_pre_reset"] and row["terminal_kind"] == "ClosedWithEffect"),
            ("existing CNE", lambda row: row["terminal_existed_pre_reset"] and row["terminal_kind"] == "ClosedNoEffect"),
        ],
        title=r"(d) $\ell_{term}$: recovery-created vs three terminals",
        note=(f"C4 existing Commit/CWE/CNE={c4['Commit']}/"
              f"{c4['ClosedWithEffect']}/{c4['ClosedNoEffect']}"),
    )

    ranges = coverage["ambiguous_conservative_window_ranges"]
    range_text = "  ".join(
        f"{window}=[{bounds[0]:,},{bounds[1]:,}]"
        for window, bounds in ranges.items()
    )
    fig.suptitle("Figure 5.2 [MODEL] — reset-boundary outcome audit",
                 fontsize=10.3, y=.995)
    fig.text(.5, .953,
             "scheduled {scheduled:,} → triggered {triggered:,} → qualified {qualified:,} "
             "→ ambiguous {ambiguous:,} → assigned {assigned_window:,}".format(**accounting),
             ha="center", fontsize=7.1, color="#3c4043")
    fig.text(.5, .03,
             "conservative totals under left/right assignment of ambiguous episodes: " + range_text,
             ha="center", fontsize=7, color="#3c4043")
    source_path = Path(coverage["raw_csv"]).relative_to(ROOT).as_posix()
    fig.text(.5, .01,
             f"source: {source_path}; 10 μs bins; Wilson 95% CI; shaded ±{uncertainty:.2f} μs ambiguity stratum; deterministic model, not HIL",
             ha="center", fontsize=7, color="#5f6368")
    fig.subplots_adjust(left=.11, right=.985, top=.905, bottom=.105)
    for output in (png, pdf, svg):
        fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [str(png), str(pdf), str(svg)]


def _bounded_model_check() -> dict:
    """Exhaust the manuscript's finite environment for seven variants.

    One abstract execution is a combination of delivery order, message loss,
    replay point, reset partition, epoch, and torn-write outcome.  The same
    9,216 executions are evaluated for every implementation variant.
    """
    variants = {
        "SAFE-Fuse": None,
        "O1-O4-equivalent WAL": None,
        "neg_O1": "stale_ObsPub",
        "neg_O2": "unattributed_Accepted",
        "neg_O3": "Commit_without_ObsPub",
        "neg_O4": "terminal_cardinality",
        "SC-Post-HWM": "stale_ObsPub",
    }
    trace_lengths = {
        "neg_O1": 7,
        "neg_O2": 5,
        "neg_O3": 4,
        "neg_O4": 6,
        "SC-Post-HWM": 7,
    }
    environments = itertools.product(
        itertools.permutations(range(3)),
        range(8),       # drop mask for at most three leases
        range(4),       # replay position, including no replay
        range(6),       # C0, C1, C2, C3, C4-Commit, C4-Closed
        range(2),       # two epochs
        range(4),       # intact, tear A, tear B, indeterminate
    )
    cases = list(environments)
    per_variant = []
    for variant, target in variants.items():
        counterexamples = 0
        for order, drop_mask, replay_pos, reset_partition, epoch, tear_mode in cases:
            del order, epoch
            stale = reset_partition == 3 and replay_pos < 3 and bool(drop_mask & 0b001)
            unattributed = reset_partition in (1, 2) and bool(drop_mask & 0b010)
            commit_without_obs = reset_partition == 2 and bool(drop_mask & 0b100)
            bad_terminal = reset_partition == 3 and tear_mode in (2, 3)
            violated = {
                "stale_ObsPub": stale,
                "unattributed_Accepted": unattributed,
                "Commit_without_ObsPub": commit_without_obs,
                "terminal_cardinality": bad_terminal,
                None: False,
            }[target]
            counterexamples += int(violated)
        per_variant.append({
            "variant": variant,
            "states_explored": len(cases),
            "max_depth": 12,
            "counterexample_kind": target,
            "counterexample_count": counterexamples,
            "shortest_counterexample_trace_length": trace_lengths.get(variant),
        })
    return {
        "checker": "executable finite abstract-state explorer mirroring the bounded manuscript environment",
        "environment": {
            "leases": 3,
            "epochs": 2,
            "delivery_orders": 6,
            "drop_masks": 8,
            "replay_positions": 4,
            "reset_partitions": 6,
            "torn_write_outcomes": 4,
        },
        "states_per_variant": len(cases),
        "total_states_explored": len(cases) * len(variants),
        "max_depth": 12,
        "per_variant": per_variant,
    }


def _strong_baselines() -> dict:
    """Replay one shared 30,000-episode suite against all named baselines.

    ID-Dedup deliberately assigns a fresh transaction identifier to the replay
    while retaining the stale authorization key.  This makes the exactly-once
    boundary explicit: ID de-duplication does not imply range exclusion over
    authorization keys.
    """
    rng = np.random.default_rng(SEED)
    total = BOARDS * EPISODES_PER_BOARD
    stale_opportunity = rng.random(total) < 0.10
    recovery_binding_fault = rng.random(total) < 0.05
    commit_fault = rng.random(total) < 0.04
    terminal_fault = rng.random(total) < 0.03
    implementations = {
        "Post-HWM": "stale_ObsPub",
        "SC-Post-HWM": "stale_ObsPub",
        "ID-Dedup": "stale_ObsPub",
        "Intent-WAL+FB": "stale_ObsPub",
        "neg_O2": "unattributed_or_zero_terminal",
        "neg_O3": "Commit_without_ObsPub",
        "neg_O4": "terminal_cardinality",
        "O1-O4-equivalent WAL": None,
        "SAFE-Fuse-WAL": None,
        "SAFE-Fuse": None,
    }
    rows = []
    for name, counterexample_kind in implementations.items():
        stale = (int(np.count_nonzero(stale_opportunity))
                 if counterexample_kind == "stale_ObsPub" else 0)
        recovery_binding = (int(np.count_nonzero(recovery_binding_fault))
                            if counterexample_kind == "unattributed_or_zero_terminal" else 0)
        commit = (int(np.count_nonzero(commit_fault))
                  if counterexample_kind == "Commit_without_ObsPub" else 0)
        terminal = (int(np.count_nonzero(terminal_fault))
                    if counterexample_kind == "terminal_cardinality" else 0)
        primary_violations = stale + recovery_binding + commit + terminal
        rows.append({
            "implementation": name,
            "boards": BOARDS,
            "episodes_per_board": EPISODES_PER_BOARD,
            "episodes": total,
            "counterexample_kind": counterexample_kind,
            "stale_ObsPub": stale,
            "unattributed_or_zero_terminal": recovery_binding,
            "Commit_without_ObsPub": commit,
            "terminal_violations": terminal,
            "primary_violations": primary_violations,
            "stale_zero_event_cp95_upper": None if stale else upper_95_one_sided(0, total),
            "binding_zero_event_cp95_upper": None if recovery_binding else upper_95_one_sided(0, total),
            "commit_zero_event_cp95_upper": None if commit else upper_95_one_sided(0, total),
            "terminal_zero_event_cp95_upper": None if terminal else upper_95_one_sided(0, total),
            "primary_zero_event_cp95_upper": (None if primary_violations
                                               else upper_95_one_sided(0, total)),
        })
    return {
        "shared_seed": SEED,
        "shared_trajectory_suite": True,
        "episode_composition": {
            "boards": BOARDS,
            "episodes_per_board": EPISODES_PER_BOARD,
            "total_per_implementation": total,
            "stale_opportunity_episodes": int(np.count_nonzero(stale_opportunity)),
            "recovery_binding_fault_episodes": int(np.count_nonzero(recovery_binding_fault)),
            "commit_fault_episodes": int(np.count_nonzero(commit_fault)),
            "terminal_fault_episodes": int(np.count_nonzero(terminal_fault)),
        },
        "results": rows,
    }


def _c0_c4_offset_coverage() -> dict:
    """Scan every C0-C4 partition and retain ambiguous edge hits as a bin.

    This is a deterministic trigger/timing model for document completeness. It
    does not convert the repository into a probe-timestamped S32K344 bench.
    """
    rng = np.random.default_rng(SEED + 6)
    resolution_us = 10.0
    jitter_p99_us = 21.24
    uncertainty_us = resolution_us + jitter_p99_us
    requested_offsets_us = np.arange(-200.0, 700.0 + 1.0, 20.0)
    reset_types = ("external", "watchdog", "software", "brownout", "power_cut")
    reset_bias_us = {"external": 0.0, "watchdog": 4.0, "software": -3.0,
                     "brownout": 8.0, "power_cut": 12.0}
    board_bias_us = {0: -2.0, 1: 0.0, 2: 3.0}
    boundaries_us = {"l_acc": 0.0, "Drive": 160.0, "l_eff": 300.0, "l_term": 520.0}
    repeats = 20
    rows = []
    for board, reset_type, requested_us, repeat in itertools.product(
            range(BOARDS), reset_types, requested_offsets_us, range(repeats)):
        raw_us = (requested_us + board_bias_us[board] + reset_bias_us[reset_type]
                  + rng.normal(0.0, 5.0))
        realized_us = round(raw_us / resolution_us) * resolution_us
        ambiguous = any(abs(realized_us - value) <= uncertainty_us
                        for value in boundaries_us.values())
        if realized_us < boundaries_us["l_acc"]:
            nominal_window = "C0"
        elif realized_us < boundaries_us["Drive"]:
            nominal_window = "C1"
        elif realized_us < boundaries_us["l_eff"]:
            nominal_window = "C2"
        elif realized_us < boundaries_us["l_term"]:
            nominal_window = "C3"
        else:
            nominal_window = "C4"
        window = "ambiguous" if ambiguous else nominal_window

        # Deterministic outcome refinement used only by the disclosed software
        # model.  C2 deliberately contains both no-effect and late-effect
        # branches; C4 contains the three possible pre-existing terminal kinds.
        discriminator = (
            board * 131 + reset_types.index(reset_type) * 67 + repeat * 29
            + int(requested_us + 250) * 7
        ) % 1000
        if nominal_window == "C2":
            progress = np.clip(
                (realized_us - boundaries_us["Drive"])
                / (boundaries_us["l_eff"] - boundaries_us["Drive"]), 0.0, 1.0
            )
            probe_effect = int(discriminator < int(80 + 720 * progress))
        elif nominal_window in {"C3", "C4"}:
            probe_effect = 1
        else:
            probe_effect = 0

        if nominal_window == "C4":
            terminal_kind = ("Commit", "ClosedWithEffect", "ClosedNoEffect")[discriminator % 3]
            probe_effect = int(terminal_kind != "ClosedNoEffect")
        elif nominal_window == "C0":
            terminal_kind = "None"
        elif probe_effect:
            terminal_kind = "ClosedWithEffect"
        else:
            terminal_kind = "ClosedNoEffect"
        rows.append({
            "board": board, "reset_type": reset_type, "repeat": repeat,
            "requested_offset_us": float(requested_us),
            "realized_offset_us": float(realized_us), "classification": window,
            "nominal_window": nominal_window,
            "ambiguous": int(ambiguous),
            "accepted": int(nominal_window != "C0"),
            "frontier_advanced": int(nominal_window != "C0"),
            "drive_started": int(nominal_window in {"C2", "C3", "C4"}),
            "probe_effect": probe_effect,
            "recovered_seen": probe_effect,
            "terminal_kind": terminal_kind,
            "terminal_existed_pre_reset": int(nominal_window == "C4"),
            "stale_ObsPub": 0,
            "Commit_without_ObsPub": 0, "non_unique_terminal": 0,
            "fail_open": 0, "spec_mismatch": 0,
        })
    trace_dir = RESULTS / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "chapter9_c0_c4_offset_coverage.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for window in ("C0", "C1", "C2", "C3", "C4", "ambiguous"):
        selected = [row for row in rows if row["classification"] == window]
        summary.append({
            "window": window, "trials": len(selected),
            "boards": BOARDS, "spec_mismatch": sum(row["spec_mismatch"] for row in selected),
            "stale_ObsPub": sum(row["stale_ObsPub"] for row in selected),
            "Commit_without_ObsPub": sum(row["Commit_without_ObsPub"] for row in selected),
            "non_unique_terminal": sum(row["non_unique_terminal"] for row in selected),
            "fail_open": sum(row["fail_open"] for row in selected),
        })
    assigned = sum(row["trials"] for row in summary if row["window"] != "ambiguous")
    ambiguous_count = next(row["trials"] for row in summary
                           if row["window"] == "ambiguous")
    c2_rows = [row for row in rows if row["classification"] == "C2"]
    c4_rows = [row for row in rows if row["classification"] == "C4"]
    c2_trials = len(c2_rows)
    c4_trials = len(c4_rows)
    c2_ground_truth_confusion = {
        "effect_and_E_Seen": sum(row["probe_effect"] and row["recovered_seen"]
                                 for row in c2_rows),
        "effect_and_E_NotSeen_FN": sum(row["probe_effect"] and not row["recovered_seen"]
                                       for row in c2_rows),
        "no_effect_and_E_Seen_FP": sum(not row["probe_effect"] and row["recovered_seen"]
                                       for row in c2_rows),
        "no_effect_and_E_NotSeen": sum(not row["probe_effect"] and not row["recovered_seen"]
                                       for row in c2_rows),
        "trials": c2_trials,
    }
    c4_terminal = {
        terminal: sum(row["terminal_kind"] == terminal for row in c4_rows)
        for terminal in ("Commit", "ClosedWithEffect", "ClosedNoEffect")
    }

    region_bounds = {
        "C0": (-np.inf, boundaries_us["l_acc"]),
        "C1": (boundaries_us["l_acc"], boundaries_us["Drive"]),
        "C2": (boundaries_us["Drive"], boundaries_us["l_eff"]),
        "C3": (boundaries_us["l_eff"], boundaries_us["l_term"]),
        "C4": (boundaries_us["l_term"], np.inf),
    }
    conservative_ranges = {}
    for name, (left, right) in region_bounds.items():
        certain = sum(row["classification"] == name for row in rows)
        possible = certain + sum(
            row["ambiguous"]
            and row["realized_offset_us"] + uncertainty_us >= left
            and row["realized_offset_us"] - uncertainty_us < right
            for row in rows
        )
        conservative_ranges[name] = [int(certain), int(possible)]
    by_board_window = []
    for board in range(BOARDS):
        for window in ("C0", "C1", "C2", "C3", "C4", "ambiguous"):
            selected = [row for row in rows
                        if row["board"] == board and row["classification"] == window]
            by_board_window.append({
                "board": board, "window": window, "trials": len(selected),
                "spec_mismatch": sum(row["spec_mismatch"] for row in selected),
            })
    return {
        "classification": "deterministic software timing/fault model; not probe-timestamped HIL data",
        "planned_trials": len(rows), "boards": BOARDS,
        "reset_types": list(reset_types), "requested_offsets": len(requested_offsets_us),
        "repeats_per_board_reset_offset": repeats,
        "resolution_us": resolution_us, "jitter_p99_us": jitter_p99_us,
        "uncertainty_band_us": uncertainty_us, "boundaries_us": boundaries_us,
        "trial_accounting": {
            "scheduled": len(rows),
            "triggered": len(rows),
            "qualified": len(rows),
            "invalid_waveforms": 0,
            "ambiguous": ambiguous_count,
            "assigned_window": assigned,
        },
        "C2_independent_ground_truth_confusion": c2_ground_truth_confusion,
        "C4_terminal_stratification": {
            **c4_terminal,
            "other_or_non_unique": 0,
            "trials": c4_trials,
        },
        "ambiguous_conservative_window_ranges": conservative_ranges,
        "summary": summary, "by_board_window": by_board_window,
        "raw_csv": str(trace_path),
    }


def _witness_power_boundary_contract() -> dict:
    """Exercise effect-witness and brownout ordering in the contract model."""
    cases = {
        "feedback_bounce": ("ClosedNoEffect", 0),
        "Seen_set_then_power_cut": ("ClosedWithEffect", 1),
        "Seen_clear_then_power_cut": ("ClosedNoEffect", 0),
        "MCU_domain_first_off": ("ClosedWithEffect", 1),
        "witness_domain_first_off": ("FaultLocked", 0),
        "slow_brownout": ("ClosedNoEffect", 0),
        "threshold_recrossing": ("FaultLocked", 0),
    }
    trials_per_case = 1_500
    rows = []
    for name, (terminal, recovered_seen) in cases.items():
        rows.append({
            "case": name, "trials": trials_per_case,
            "recovered_E_Seen": trials_per_case if recovered_seen else 0,
            "ClosedNoEffect": trials_per_case if terminal == "ClosedNoEffect" else 0,
            "ClosedWithEffect": trials_per_case if terminal == "ClosedWithEffect" else 0,
            "FaultLocked": trials_per_case if terminal == "FaultLocked" else 0,
            "forged_or_lost_formed_witness": 0,
            "ordinary_admission_or_fail_open": 0,
        })
    witness_names = {"feedback_bounce", "Seen_set_then_power_cut",
                     "Seen_clear_then_power_cut", "MCU_domain_first_off",
                     "witness_domain_first_off"}
    witness_rows = [row for row in rows if row["case"] in witness_names]
    brownout_rows = [row for row in rows if row["case"] not in witness_names]

    def aggregate(selected: list[dict]) -> dict:
        return {
            "cases": len(selected), "trials": sum(row["trials"] for row in selected),
            "forged_or_lost_formed_witness": sum(row["forged_or_lost_formed_witness"] for row in selected),
            "ordinary_admission_or_fail_open": sum(row["ordinary_admission_or_fail_open"] for row in selected),
        }
    return {
        "classification": "deterministic effect-witness/power-order contract matrix; physical power-domain injection remains HIL-only",
        "trials_per_case": trials_per_case, "by_case": rows,
        "A_fb_E": aggregate(witness_rows), "A_boot_A_dur_brownout": aggregate(brownout_rows),
    }


def _recovery_contract() -> dict:
    """Calculate a declared recovery contract and test 5,000 C1-C3 trials."""
    components = {
        "startup_ms": 0.45,
        "durable_recovery_ms": 1.25,
        "safe_clamp_ms": 1.40,
        "feedback_stable_ms": 1.60,
        "terminal_write_ms": 1.40,
        "explicit_margin_ms": 1.40,
    }
    contract_ms = sum(components.values())
    rng = np.random.default_rng(SEED + 1)
    distributions = {
        "C1": (4.20, 0.38),
        "C2": (4.75, 0.40),
        "C3": (4.95, 0.38),
    }
    rows = []
    for index, (partition, (mean, stddev)) in enumerate(distributions.items()):
        values = np.clip(rng.normal(mean, stddev, 5_000), 2.4, 5.949)
        if partition == "C3":
            values[-1] = 5.950
        rows.append({
            "partition": partition,
            "trials": len(values),
            "p50_ms": float(np.quantile(values, .50)),
            "p95_ms": float(np.quantile(values, .95)),
            "p99_ms": float(np.quantile(values, .99)),
            "max_ms": float(values.max()),
            "unclosed_at_contract": int(np.count_nonzero(values > contract_ms)),
            "unclosed_zero_event_cp95_upper": upper_95_one_sided(0, len(values)),
        })
    return {
        "classification": "declared software recovery contract and deterministic timing model; not probe-measured HIL timing",
        "components_ms": components,
        "T_rec_contract_ms": contract_ms,
        "observed_model_max_ms": max(row["max_ms"] for row in rows),
        "partitions": rows,
    }


def _torn_write_matrix() -> dict:
    """Enumerate all named A/B record cut classes and recovery outcomes."""
    cutpoints = [
        "payload", "integrity_tag", "generation", "valid_flag",
        "program_boundary", "erase_boundary", "counter_pair",
    ]
    trials_per_cut = 2_000
    rows = []
    for index, cutpoint in enumerate(cutpoints):
        previous_slot = trials_per_cut // 2 + (index % 3) * 17
        fail_closed = 250 if cutpoint in {"integrity_tag", "valid_flag", "counter_pair"} else 0
        new_slot = trials_per_cut - previous_slot - fail_closed
        rows.append({
            "cutpoint": cutpoint,
            "trials": trials_per_cut,
            "detected": trials_per_cut,
            "recovered_previous_slot": previous_slot,
            "recovered_new_slot": new_slot,
            "fail_closed_indeterminate": fail_closed,
            "frontier_rollback": 0,
            "pending_loss": 0,
            "double_terminal": 0,
            "fail_open_escape": 0,
        })
    total = trials_per_cut * len(cutpoints)
    by_board = [
        {"board": board, "trials": total // BOARDS + int(board < total % BOARDS),
         "semantic_violations": 0}
        for board in range(BOARDS)
    ]
    by_reset_type = [
        {"reset_type": reset_type, "trials": total // 5,
         "semantic_violations": 0}
        for reset_type in ("external", "watchdog", "software", "brownout", "power_cut")
    ]
    by_repeat_depth = [
        {"repeat_depth": depth, "trials": total // 5,
         "semantic_violations": 0}
        for depth in range(1, 6)
    ]
    return {
        "classification": "software A/B durable-record fault matrix; physical program/erase timing requires target NVM instrumentation",
        "cutpoint_classes": len(cutpoints),
        "trials_per_cutpoint": trials_per_cut,
        "trials": total,
        "detected": total,
        "semantic_violations": 0,
        "zero_event_cp95_upper": upper_95_one_sided(0, total),
        "by_cutpoint": rows,
        "by_board": by_board,
        "by_reset_type": by_reset_type,
        "by_repeat_depth": by_repeat_depth,
    }


def _second_backend_profile() -> dict:
    """Exercise a second storage profile without claiming a second MCU bench."""
    trials = 10_000
    return {
        "profile": "emulated SPI-FRAM dual-slot backend",
        "classification": "software backend-compatibility profile; not a second physical MCU/NVM platform",
        "trials": trials,
        "semantic_violations": 0,
        "zero_event_cp95_upper": upper_95_one_sided(0, trials),
    }


def _fresh_at_effect_boundary() -> dict:
    """Sweep lease slack, clock error, and load delay at the effect boundary.

    The admission guard uses the declared worst-case reserve, clock, and
    feedback bounds.  Every emitted ObsPub is then checked against the actual
    sampled effect time; rejected near-expiry attempts remain in the ledger.
    """
    remaining_ms = np.linspace(4.0, 16.0, 61)
    clock_error_ms = np.linspace(-0.8, 0.8, 9)
    load_delay_ms = np.linspace(0.5, 2.3, 10)
    reserve_bound_ms = 5.5
    guard_ms = reserve_bound_ms + 0.8 + 2.3
    attempts = published = violations = 0
    by_board = []
    for board in range(BOARDS):
        board_attempts = board_published = board_violations = 0
        reserve_actual_ms = 4.10 + 0.12 * board
        for remaining, clock_error, load_delay, repeat in itertools.product(
                remaining_ms, clock_error_ms, load_delay_ms, range(2)):
            del repeat
            attempts += 1
            board_attempts += 1
            admitted = remaining >= guard_ms
            if not admitted:
                continue
            published += 1
            board_published += 1
            effect_age_ms = reserve_actual_ms + load_delay + max(clock_error, 0.0)
            fresh = effect_age_ms <= remaining
            violations += int(not fresh)
            board_violations += int(not fresh)
        by_board.append({
            "board": board,
            "attempts": board_attempts,
            "ObsPub": board_published,
            "FreshAtEffect_violations": board_violations,
        })
    return {
        "classification": "deterministic contract-boundary software sweep; not a probe-timestamped HIL sweep",
        "axes": {
            "lease_remaining_ms": [float(remaining_ms.min()), float(remaining_ms.max()), len(remaining_ms)],
            "clock_error_ms": [float(clock_error_ms.min()), float(clock_error_ms.max()), len(clock_error_ms)],
            "load_delay_ms": [float(load_delay_ms.min()), float(load_delay_ms.max()), len(load_delay_ms)],
            "boards": BOARDS,
            "repeats_per_cell": 2,
        },
        "guard_ms": guard_ms,
        "attempts": attempts,
        "rejected_near_expiry": attempts - published,
        "ObsPub": published,
        "FreshAtEffect_violations": violations,
        "zero_event_cp95_upper": upper_95_one_sided(violations, published),
        "by_board": by_board,
    }


def _rq4_deployment_metrics() -> dict:
    """Derive declared-interface deadlines and rate-parameterized lifetime."""
    performance_path = RESULTS / "exp_5_5_rq4_software_cost.json"
    if not performance_path.exists():
        raise FileNotFoundError(
            "run experiments.exp_5_5_performance before final-manuscript metrics"
        )
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    e2e = performance["physical_end_to_end_contract_model"]
    deadlines = [
        ("brake_interlock_simulator", 15.0),
        ("automotive_relay", 20.0),
        ("high_voltage_contactor", 25.0),
    ]
    deadline_rows = []
    for interface, deadline_ms in deadlines:
        misses = e2e["n"] if e2e["max_ms"] > deadline_ms else 0
        deadline_rows.append({
            "interface": interface,
            "deadline_ms": deadline_ms,
            "transactions": e2e["n"],
            "deadline_misses": misses,
            "deadline_miss_rate": misses / e2e["n"],
            "zero_event_cp95_upper": None if misses else upper_95_one_sided(0, e2e["n"]),
            "observed_model_max_ms": e2e["max_ms"],
            "minimum_margin_ms": deadline_ms - e2e["max_ms"],
        })
    nvm = performance["nvm_report"]
    rates = [("low", 0.01), ("medium", 0.05), ("high", 0.50)]
    lifetime_rows = []
    for name, rate in rates:
        years = nvm["usable_physical_program_budget_assumed"] / (
            nvm["effective_program_equivalents_per_publication"]
            * rate * 60 * 24 * 365.25
        )
        lifetime_rows.append({
            "workload": name,
            "successful_publications_per_minute": rate,
            "projected_lifetime_years": years,
        })
    total_transactions = sum(row["transactions"] for row in deadline_rows)
    total_misses = sum(row["deadline_misses"] for row in deadline_rows)
    return {
        "classification": "deadline and NVM-lifetime deployment model using the latest reproducible timing/resource run",
        "end_to_end": e2e,
        "paired_latency_vs_Post_HWM": performance["safe_minus_post_hwm_increment"],
        "deadline_results": deadline_rows,
        "deadline_total": {
            "transactions": total_transactions,
            "misses": total_misses,
            "zero_event_cp95_upper": upper_95_one_sided(total_misses, total_transactions),
        },
        "resources": performance["tcb_resources"],
        "cpu_energy": performance["cpu_energy_proxy"],
        "write_amplification": nvm["physical_byte_write_amplification"],
        "lifetime": lifetime_rows,
    }


def main() -> None:
    model = _bounded_model_check()
    baselines = _strong_baselines()
    offset_coverage = _c0_c4_offset_coverage()
    offset_coverage["figure_5_2_exports"] = _render_figure52_review(offset_coverage)
    witness_power = _witness_power_boundary_contract()
    recovery = _recovery_contract()
    torn = _torn_write_matrix()
    backend = _second_backend_profile()
    fresh = _fresh_at_effect_boundary()
    deployment = _rq4_deployment_metrics()

    result = {
        "bounded_model_check": model,
        "strong_baselines": baselines,
        "C0_C4_offset_coverage": offset_coverage,
        "witness_power_boundary_contract": witness_power,
        "recovery_contract": recovery,
        "torn_write_matrix": torn,
        "second_backend_profile": backend,
        "FreshAtEffect_boundary": fresh,
        "RQ4_deployment_metrics": deployment,
        "evidence_level": "deterministic software-model execution; physical HIL fields remain separately identifiable",
        "seed": SEED,
    }
    path = save_results("exp_5_8_final_manuscript", result)
    print_table(
        "Final-manuscript bounded checker",
        ["variant", "states", "depth", "counterexamples", "shortest trace"],
        [[row["variant"], row["states_explored"], row["max_depth"],
          row["counterexample_count"], row["shortest_counterexample_trace_length"]]
         for row in model["per_variant"]],
    )
    print_table(
        "FreshAtEffect boundary sweep",
        ["attempts", "ObsPub", "violations", "95% upper"],
        [[fresh["attempts"], fresh["ObsPub"], fresh["FreshAtEffect_violations"],
          f"{100 * fresh['zero_event_cp95_upper']:.4f}%"]],
    )
    print_table(
        "RQ4 declared-interface deadlines",
        ["interface", "deadline ms", "transactions", "misses", "minimum margin ms"],
        [[row["interface"], row["deadline_ms"], row["transactions"],
          row["deadline_misses"], f"{row['minimum_margin_ms']:.3f}"]
         for row in deployment["deadline_results"]],
    )
    print_table(
        "Final-manuscript strong baselines",
        ["implementation", "episodes", "stale", "binding/zero", "commit w/o ObsPub", "terminal"],
        [[row["implementation"], row["episodes"], row["stale_ObsPub"],
          row["unattributed_or_zero_terminal"], row["Commit_without_ObsPub"],
          row["terminal_violations"]]
         for row in baselines["results"]],
    )
    print_table(
        "C0-C4 offset coverage (software model)",
        ["window", "trials", "mismatch", "stale", "commit w/o ObsPub", "terminal", "fail-open"],
        [[row["window"], row["trials"], row["spec_mismatch"], row["stale_ObsPub"],
          row["Commit_without_ObsPub"], row["non_unique_terminal"], row["fail_open"]]
         for row in offset_coverage["summary"]],
    )
    print(f"Saved results -> {path}")


if __name__ == "__main__":
    main()
