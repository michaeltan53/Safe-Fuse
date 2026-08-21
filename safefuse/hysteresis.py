"""Dual-threshold hysteresis engine (§3.4 / §3.5).

The engine maps a calibrated score p_t together with the previous active
authorization a_{t-1} to a new active authorization in {ACCEPT, REJECT}. The
hysteresis half-bandwidth δ must exceed ε^+ (Theorem 1's hypothesis).
"""

from __future__ import annotations

from dataclasses import dataclass

from .state import AuthState


@dataclass
class HysteresisParams:
    theta_H: float = 0.65    # upper threshold — drop to REJECT above
    theta_L: float = 0.35    # lower threshold — climb to ACCEPT below
    # half-bandwidth δ in the paper. Must satisfy δ > ε^+ (Theorem 1).
    delta: float = 0.25


class Hysteresis:
    def __init__(self, params: HysteresisParams | None = None,
                 init_state: AuthState = AuthState.ACCEPT):
        self.p = params or HysteresisParams()
        if not (self.p.theta_L < self.p.theta_H):
            raise ValueError("theta_L must be strictly below theta_H")
        self._state: AuthState = init_state

    @property
    def state(self) -> AuthState:
        return self._state

    def reset(self, init_state: AuthState = AuthState.ACCEPT) -> None:
        self._state = init_state

    def step(self, score: float) -> AuthState:
        """Update and return the active authorization.

        Score semantic: higher == more "unsafe" / reject-leaning. Above θ_H
        forces REJECT, below θ_L forces ACCEPT, otherwise the previous active
        state persists (the hysteresis band).
        """
        if score >= self.p.theta_H:
            self._state = AuthState.REJECT
        elif score <= self.p.theta_L:
            self._state = AuthState.ACCEPT
        # else: keep self._state — this is the hysteresis hold
        return self._state
