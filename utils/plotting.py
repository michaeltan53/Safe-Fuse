"""Figure generators for paper §5 (Figures 5.1 – 5.4).

All figures render headless (Agg) into results/figs/. Imports are lazy so a
machine without matplotlib can still run the numeric experiments.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _ecdf(values):
    import numpy as np
    arr = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, arr.size + 1) / arr.size
    return arr, y


# ---------------------------------------------------------------------------
# Figure 5.1 — paired-design violation multi-panel (strong-substitute family)
# ---------------------------------------------------------------------------


def fig_5_1_violation_panels(
    order: Sequence[str],
    inversion_per_run: Dict[str, Sequence[int]],
    pseudo_per_run: Dict[str, Sequence[int]],
    out_path: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _ensure_dir(out_path)
    palette = ["#d93025", "#f9ab00", "#a142f4", "#e8710a", "#188038"]
    colors = {b: palette[i % len(palette)] for i, b in enumerate(order)}
    rng = np.random.default_rng(7)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, data, title, ylab in (
        (axes[0], inversion_per_run, "(a) semantic inversion per run",
         "semantic inversions / run"),
        (axes[1], pseudo_per_run, "(b) pseudo-trajectory extension per run",
         "pseudo-extensions / run"),
    ):
        for i, b in enumerate(order):
            vals = np.asarray(data[b], dtype=float)
            jitter = rng.uniform(-0.16, 0.16, size=vals.size)
            ax.scatter(np.full(vals.size, i) + jitter, vals, s=8, alpha=0.4,
                       color=colors[b], edgecolors="none", zorder=2)
            ax.boxplot(vals, positions=[i], widths=0.5, showfliers=False,
                       medianprops=dict(color="black"),
                       boxprops=dict(color="#5f6368"),
                       whiskerprops=dict(color="#5f6368"),
                       capprops=dict(color="#5f6368"), zorder=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, fontsize=9, rotation=12)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=-max(1, ax.get_ylim()[1] * 0.03))
    fig.suptitle("Fig. 5.1  Paired-design semantic-violation panels under "
                 "composite adversarial injection (Full hugs the 0 baseline)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5.2 — microsecond power-cut window recovery 2D heatmap
# ---------------------------------------------------------------------------


def fig_5_2_recovery_heatmap(
    grid: List[List[int]], phases: Sequence[str], window_steps: int,
    out_path: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    import numpy as np

    _ensure_dir(out_path)
    arr = np.asarray(grid)              # rows=window, cols=phase; values 0/1
    cmap = ListedColormap(["#aecbfa", "#188038"])  # normal-archive, safe-lock

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=1, origin="lower")
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels(phases)
    yt = np.linspace(0, window_steps - 1, 6)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{(v+1)*10.0/window_steps:.1f}" for v in yt])
    ax.set_xlabel("critical transition phase")
    ax.set_ylabel("power-cut delay window (μs)")
    ax.set_title("Fig. 5.2  Power-cut window recovery convergence "
                 "(only safe outcomes; no undefined state)")
    ax.legend(handles=[Patch(color="#aecbfa", label="normal-archive (safe)"),
                       Patch(color="#188038", label="fail-closed safe-lock")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
              frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5.3 — GapCert confusion matrix + FRR availability response curve
# ---------------------------------------------------------------------------


def fig_5_3_gapcert(
    classes: Sequence[str], confusion: List[List[int]],
    loss_rates: Sequence[float], frr_without: Sequence[float],
    frr_with: Sequence[float], out_path: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _ensure_dir(out_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # left: confusion matrix (rows=true class, cols=decision Accept/Reject)
    cm = np.asarray(confusion, dtype=float)
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Accept", "Reject"])
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(c, r, f"{int(cm[r, c])}", ha="center", va="center",
                    color="white" if cm[r, c] > cm.max() / 2 else "#202124",
                    fontsize=9)
    ax.set_xlabel("parser decision")
    ax.set_ylabel("policy-projection case class")
    ax.set_title("(a) GapCert semantic-filter confusion (FAR = 0)", fontsize=10)

    # right: FRR vs loss rate
    ax = axes[1]
    lr = np.asarray(loss_rates) * 100
    ax.plot(lr, np.asarray(frr_without) * 100, "o-", color="#d93025",
            lw=2, label="GapCert disabled")
    ax.plot(lr, np.asarray(frr_with) * 100, "s-", color="#188038",
            lw=2, label="GapCert enabled")
    ax.set_xlabel("legal gap / packet-loss rate (%)")
    ax.set_ylabel("false-reject rate FRR (%)")
    ax.set_title("(b) availability frontier under adverse loss", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.suptitle("Fig. 5.3  GapCert policy-semantic filtering & availability",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5.4 — revocation residual ECDF with τ_rev analytic envelope
# ---------------------------------------------------------------------------


def fig_5_4_residual_envelope(
    brake_residuals: Sequence[float], brake_bound: float,
    steer_residuals: Sequence[float], steer_bound: float,
    out_path: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _ensure_dir(out_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, vals, bound, title, color in (
        (axes[0], brake_residuals, brake_bound,
         "(a) brake domain (10 ms)", "#1a73e8"),
        (axes[1], steer_residuals, steer_bound,
         "(b) steer domain (5 ms)", "#0b8043"),
    ):
        x, y = _ecdf(vals)
        ax.plot(x, y, lw=2.0, color=color, label="residual exposure ECDF")
        for q, lab in ((0.95, "P95"), (0.99, "P99"), (0.999, "P99.9")):
            xv = float(np.quantile(vals, q))
            ax.scatter([xv], [q], s=28, color=color, zorder=3)
            ax.annotate(f"{lab}={xv:.1f}", (xv, q), fontsize=7,
                        xytext=(-4, -10), textcoords="offset points")
        ax.axvline(bound, color="#d93025", ls="--", lw=1.8,
                   label=f"τ_rev = {bound:.2f} ms")
        ax.fill_betweenx([0, 1], bound, bound * 1.15, color="#fce8e6",
                         zorder=0)
        ax.set_xlabel("revocation residual exposure (ms)")
        ax.set_ylabel("cumulative probability")
        ax.set_xlim(0, bound * 1.12)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Fig. 5.4  Revocation residual-exposure ECDF rigidly bounded "
                 "by the τ_rev analytic envelope (Bound Slack ≥ 0)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
