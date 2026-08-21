"""Operational-domain gates Ω_score / Ω_step / Ω_manifold / Ω_noise (§3.3).

The four gates jointly define the auditable operational domain Ω_n^+ on which
Theorem 1 holds. They are evaluated only after `ValidEvidence_t` is true
(§3.4) so we never touch undefined state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math


@dataclass
class DomainParams:
    """Conservative offline-calibrated bounds (Table 5.2)."""

    L_P_plus: float = 2.0           # score-regularity upper bound
    eps_plus: float = 0.05          # noise residual upper bound ε^+
    eta_plus: float = 15.6          # manifold divergence threshold η^+
    delta_max_plus: float = 0.05    # per-step physical bound Δ_max^+


@dataclass
class GateOutcome:
    in_domain: bool
    fail_reason: str = ""   # one of "", "score", "step", "manifold", "noise"

    @property
    def reason_code(self) -> str:
        return {
            "score": "ERR_DOMAIN_SCORE",
            "step": "ERR_DOMAIN_STEP",
            "manifold": "ERR_DOMAIN_MANIFOLD",
            "noise": "ERR_DOMAIN_NOISE",
            "": "OK",
        }[self.fail_reason]


class OperationalDomain:
    """Stateful runtime assertion engine.

    The internal state stores only the previous step's authenticated proxy
    `Ŝ_{t-1}` and the previous calibrated score `p_{t-1}`. This is the
    minimum needed to enforce Ω_score (local Lipschitz on observed scores) and
    Ω_step (per-step physical displacement bound).
    """

    def __init__(self, params: Optional[DomainParams] = None):
        self.p = params or DomainParams()
        self._prev_score: Optional[float] = None
        self._prev_proxy: Optional[float] = None

    def reset(self) -> None:
        self._prev_score = None
        self._prev_proxy = None

    # ---- individual gates -------------------------------------------------

    def _omega_step(self, s_t: float) -> bool:
        if self._prev_proxy is None:
            return True   # Ω_init absorbs the t=1 case
        return abs(s_t - self._prev_proxy) <= self.p.delta_max_plus

    def _omega_score(self, p_t: float, s_t: float) -> bool:
        if self._prev_score is None or self._prev_proxy is None:
            return True
        rhs = self.p.L_P_plus * abs(s_t - self._prev_proxy) + 2.0 * self.p.eps_plus
        return abs(p_t - self._prev_score) <= rhs

    def _omega_manifold(self, manifold_divergence: float) -> bool:
        return manifold_divergence <= self.p.eta_plus

    def _omega_noise(self, residual_estimate: float) -> bool:
        return residual_estimate <= self.p.eps_plus

    # ---- composite gate ---------------------------------------------------

    def evaluate(
        self,
        score: float,
        proxy: float,
        manifold_divergence: float,
        residual_estimate: float,
    ) -> GateOutcome:
        """Evaluate Ω^(t). Order matches §3.3 listing.

        Returns the first failing gate so the audit log can identify which
        physical bound was breached. State is updated only on success — a
        failure cleanly enters F without contaminating the priors used by
        Theorem 1.
        """

        if not self._omega_noise(residual_estimate):
            return GateOutcome(False, "noise")
        if not self._omega_manifold(manifold_divergence):
            return GateOutcome(False, "manifold")
        if not self._omega_step(proxy):
            return GateOutcome(False, "step")
        if not self._omega_score(score, proxy):
            return GateOutcome(False, "score")

        self._prev_score = score
        self._prev_proxy = proxy
        return GateOutcome(True, "")


def tau_d(p: DomainParams, delta_hysteresis: float) -> int:
    """Theorem 4 minimum dwell time τ_d = ⌈ 2δ / (L_P^+ Δ_max^+ + 2ε^+) ⌉.

    Returns ∞ (encoded as a very large int) when the non-trivial residency
    condition L_P^+ Δ_max^+ + 2ε^+ < 2δ fails — see §3.3 edge cases.
    """
    den = p.L_P_plus * p.delta_max_plus + 2.0 * p.eps_plus
    if den <= 0 or 2.0 * delta_hysteresis <= 0:
        return 10**9
    return int(math.ceil(2.0 * delta_hysteresis / den))


def bounded_switch_bound(n_steps: int, tau: int) -> int:
    """Theorem 4 analytic upper bound:
    N_term ≤ ⌊(|I| − 1)/τ_d⌋ + 2,  with |I|·=·n_steps."""
    if n_steps <= 0 or tau <= 0:
        return 0
    if n_steps == 1:
        return 1
    return (n_steps - 1) // tau + 2
