"""The four local-violation predicates of paper §5.2.

Each scanner is a pure function over an ordered log `L` (a list of PubEvent)
running in O(|L|). They never make subjective inferences — a violation is a
purely syntactic property of the published trace.

    SemanticInversion(L) := ∃ e1,e2 ∈ L .
        t_publish(e2) < t_publish(e1) ∧ (e1.epoch,e1.seq) < (e2.epoch,e2.seq)

    LateUnsafeAccept(L) := ∃ e ∈ L .
        t_publish(e) > t_revoke + τ_rev^impl ∧ is_active(e)

    SwitchOverflow(J) := N_flip(J) > ⌊|J| / Δ_dwell⌋ + B⁺

    PublishCommitGap(L) := ∃ t_publish .
        ¬∃ t_commit ∈ [t_publish − δ_scan^max, t_publish]
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .publisher import PubEvent
from .state import AuthState


def semantic_inversion(trace: Sequence[PubEvent]) -> int:
    """Count of (publish-order, logical-order) inversions.

    The pin publishes in `t_publish` order; a violation is any pair whose
    logical key (epoch, lease_seq) regresses while publish time advances.
    Because publication is already time-ordered, this reduces to counting
    descents of the logical key along the trace.
    """
    inversions = 0
    if not trace:
        return 0
    ordered = sorted(trace, key=lambda e: e.t_publish_ms)
    hi = (-1, -1)
    for e in ordered:
        key = (e.epoch, e.lease_seq)
        if key < hi:
            inversions += 1
        else:
            hi = key
    return inversions


def has_semantic_inversion(trace: Sequence[PubEvent]) -> bool:
    return semantic_inversion(trace) > 0


def late_unsafe_accept(
    trace: Sequence[PubEvent], t_revoke_ms: float, tau_rev_impl_ms: float
) -> int:
    """Active states published later than t_revoke + τ_rev^impl."""
    deadline = t_revoke_ms + tau_rev_impl_ms
    return sum(1 for e in trace if e.t_publish_ms > deadline and e.is_active)


def switch_overflow(
    n_flip: int, window_len: int, dwell: int, b_plus: int
) -> bool:
    """True if the active-state flip count exceeds the analytic bound."""
    return n_flip > (window_len // max(1, dwell)) + b_plus


def publish_commit_gap(
    trace: Sequence[PubEvent], delta_scan_ms: float
) -> int:
    """Count of publishes with no commit inside [t_publish−δ_scan, t_publish]."""
    gaps = 0
    for e in trace:
        if not (e.t_publish_ms - delta_scan_ms <= e.t_commit_ms <= e.t_publish_ms):
            gaps += 1
    return gaps


def residual_exposure_ms(
    trace: Sequence[PubEvent], t_revoke_ms: float
) -> float:
    """Largest publish time of an active state after t_revoke, minus t_revoke.
    This is the physically observable stale residual on the pin."""
    actives = [e.t_publish_ms for e in trace
               if e.is_active and e.t_publish_ms >= t_revoke_ms]
    if not actives:
        return 0.0
    return max(actives) - t_revoke_ms
