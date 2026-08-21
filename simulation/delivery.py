"""Untrusted-bus delivery adversary and SAFE-Fuse AV harness (paper §5.3).

The attacker operates purely at the delivery layer: it never breaks the
cryptographic primitives or tampers with payloads. It can only

    * reorder delivered lease frames,
    * hold the high-water (newest) frame back, and
    * release a stale active frame late,

exactly the §5.3.1 threat model. We build genuinely-signed leases so the
SAFE-Fuse actuator verifier (AV) really verifies + applies its high-water
mark; the AV's zero-inversion result is therefore a mechanism check, not a
calibrated constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from safe_fuse.crypto import SigningKey, sha256
from safe_fuse.domains import DomainConfig
from safe_fuse.lease import Lease, issue_lease
from safe_fuse.publisher import ActuatorVerifier
from safe_fuse.predicates import semantic_inversion, residual_exposure_ms
from safe_fuse.state import AuthState


@dataclass
class DeliveredLease:
    lease: Lease
    arrival_ms: float
    counter: int


class DeliveryHarness:
    """Builds signed lease windows and replays attacker-reordered deliveries."""

    def __init__(self, domain: DomainConfig, seed: int = 0):
        self.domain = domain
        self._sk = SigningKey(seed=bytes([seed & 0xFF]) * 32)
        self.vk = self._sk.verifying_key()
        self.domain_id = ("safe-fuse:dom:" + domain.name).encode()
        self._epoch = 1
        self._seq = 0
        self._counter = 0
        self._t = 1000.0

    def _issue(self, state: AuthState) -> Lease:
        self._seq += 1
        self._counter += 1
        d = sha256(self._seq.to_bytes(8, "big"))
        lease = issue_lease(
            self._sk,
            domain=self.domain_id, state=state,
            digest_d=d, meta_hash=d,
            c_start=self._counter, c_end=self._counter + self.domain.w_max,
            exp_ms=int(self._t + self.domain.t_lease_ms),
            epoch=self._epoch, head=d, err="OK",
            keyver=1, lease_seq=self._seq,
        )
        return lease

    def reorder_window(
        self, n_pairs: int, rng: np.random.Generator
    ) -> List[DeliveredLease]:
        """One stress window. Emits `n_pairs` (Active, newer-Revoke) genesis
        pairs but delivers each pair reversed (Revoke first, stale Active
        held and released late) — a post-delivery semantic inversion attempt.
        """
        delivered: List[DeliveredLease] = []
        for _ in range(n_pairs):
            self._t += self.domain.delta_auth_ms
            active = self._issue(AuthState.ACCEPT)
            revoke = self._issue(AuthState.REJECT)
            hold = float(rng.uniform(self.domain.delta_auth_ms,
                                     4 * self.domain.delta_auth_ms))
            # bus delivers the newer Revoke first…
            delivered.append(DeliveredLease(revoke, self._t, revoke.c_start))
            # …then releases the stale Active later (the attack).
            delivered.append(DeliveredLease(active, self._t + hold,
                                            active.c_start))
        return delivered

    def run_safe_fuse_av(
        self, delivered: List[DeliveredLease]
    ) -> Tuple[int, float]:
        """Run the genuine SAFE-Fuse AV (high-water + commit-before-publish)
        over the delivered stream. Returns (inversions, max_stale_residual)."""
        av = ActuatorVerifier(
            vk=self.vk, domain_id=self.domain_id,
            tau_rev_impl_ms=self.domain.tau_rev_impl_ms,
            delta_scan_ms=self.domain.delta_scan_ms,
            t_fb_ms=self.domain.t_fb_ms,
        )
        # deliver in arrival-time order (the attacker's chosen order)
        for d in sorted(delivered, key=lambda x: x.arrival_ms):
            av.deliver(d.lease, now_ms=d.arrival_ms,
                       current_counter=d.counter)
        inv = semantic_inversion(av.pub_trace)
        # residual measured from the first Revoke publish
        revokes = [e.t_publish_ms for e in av.pub_trace
                   if e.state == AuthState.REJECT]
        residual = 0.0
        if revokes:
            residual = residual_exposure_ms(av.pub_trace, min(revokes))
        return inv, residual
