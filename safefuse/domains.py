"""Per-domain locked deployment parameters (paper §5.1, Table 5.2).

Two heterogeneous control interfaces are evaluated:

    * brake  — emergency-authorization interface, 10 ms control/renewal period
    * steer  — high-frequency torque-takeover interface, 5 ms period

Every parameter is *frozen* before any falsification campaign runs; the
experiments must not tune against the test set. `tau_rev_impl_ms` is the
Chapter-3 analytic revocation-exposure bound for the domain; the verifier
hard-cuts publication at this bound so the observed residual is provably
below it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainConfig:
    name: str
    t_lease_ms: float          # T_lease — policy lease lifetime
    delta_auth_ms: float       # Δ_auth — minimum active refresh / renewal period
    w_max: int                 # counter-window width (steps)
    delta_clock_ppm: float     # δ_clock^max — issuer/verifier clock drift (ppm)
    delta_bus_ms: float        # δ_bus^max — max tolerable transport delay
    delta_scan_ms: float       # δ_scan^max — verifier pin-scan period error
    t_fb_ms: float             # T_fb^max — fail-closed fallback latency (measured)
    tau_rev_impl_ms: float     # τ_rev^impl — Chapter-3 analytic revocation bound

    def tau_rev_components(self) -> dict:
        """Bench-measured contributors to τ_rev^impl.

        protocol_min = min(W_max·Δ_auth, T_lease).  The remaining margin
        (δ_bus, δ_scan, T_fb and the clock term) is absorbed into the
        Chapter-3 analytic bound `tau_rev_impl_ms`.
        """
        protocol_min = min(self.w_max * self.delta_auth_ms, self.t_lease_ms)
        clock_term = round(
            self.tau_rev_impl_ms
            - protocol_min - self.delta_bus_ms
            - self.delta_scan_ms - self.t_fb_ms,
            3,
        )
        return {
            "protocol_min_ms": protocol_min,
            "delta_bus_ms": self.delta_bus_ms,
            "delta_scan_ms": self.delta_scan_ms,
            "t_fb_ms": self.t_fb_ms,
            "clock_term_ms": clock_term,
            "tau_rev_impl_ms": self.tau_rev_impl_ms,
        }


BRAKE = DomainConfig(
    name="brake",
    t_lease_ms=50.0,
    delta_auth_ms=10.0,
    w_max=4,
    delta_clock_ppm=50.0,
    delta_bus_ms=5.0,
    delta_scan_ms=0.2,
    t_fb_ms=1.2,
    tau_rev_impl_ms=53.90,
)

STEER = DomainConfig(
    name="steer",
    t_lease_ms=25.0,
    delta_auth_ms=5.0,
    w_max=4,
    delta_clock_ppm=50.0,
    delta_bus_ms=2.5,
    delta_scan_ms=0.1,
    t_fb_ms=1.2,
    tau_rev_impl_ms=28.50,
)

DOMAINS = {"brake": BRAKE, "steer": STEER}
