"""Short-lease tokens π_ℓ (§3.2 freshness, §4.6 extensions).

A lease binds a particular active authorization a_t to:
    - a domain identifier `dom_ℓ`
    - the evidence commitment q_t = H( d_t || H(m*_t) )
    - a counter window [c_start, c_end]
    - an absolute expiry `exp_t`
    - the current run-epoch g_t and audit head h_t
    - (§4.6 extension) the err code and key version

Lease verification reproduces the consumer-side check the executor would do.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .crypto import SigningKey, VerifyingKey, sha256
from .state import AuthState


@dataclass(frozen=True)
class Lease:
    domain: bytes
    state: AuthState
    q_t: bytes
    c_start: int
    c_end: int
    exp_ms: int
    epoch: int
    head: bytes
    err: str
    keyver: int
    signature: bytes
    lease_seq: int = 0          # §3 update: strict per-epoch order witness

    def message(self) -> bytes:
        return b"".join(
            [
                self.domain,
                self.state.value.encode(),
                self.q_t,
                struct.pack(">Q", self.c_start),
                struct.pack(">Q", self.c_end),
                struct.pack(">Q", self.exp_ms),
                struct.pack(">I", self.epoch),
                self.head,
                self.err.encode(),
                struct.pack(">I", self.keyver),
                struct.pack(">Q", self.lease_seq),
            ]
        )


def issue_lease(
    sk: SigningKey,
    *,
    domain: bytes,
    state: AuthState,
    digest_d: bytes,
    meta_hash: bytes,
    c_start: int,
    c_end: int,
    exp_ms: int,
    epoch: int,
    head: bytes,
    err: str,
    keyver: int,
    lease_seq: int = 0,
) -> Lease:
    q_t = sha256(digest_d + meta_hash)
    skeleton = Lease(
        domain=domain,
        state=state,
        q_t=q_t,
        c_start=c_start,
        c_end=c_end,
        exp_ms=exp_ms,
        epoch=epoch,
        head=head,
        err=err,
        keyver=keyver,
        signature=b"",
        lease_seq=lease_seq,
    )
    sig = sk.sign(skeleton.message())
    return Lease(
        domain=domain,
        state=state,
        q_t=q_t,
        c_start=c_start,
        c_end=c_end,
        exp_ms=exp_ms,
        epoch=epoch,
        head=head,
        err=err,
        keyver=keyver,
        signature=sig,
        lease_seq=lease_seq,
    )


@dataclass
class VerifierState:
    """Executor-side high-watermark for strict per-epoch linearisation."""

    cur_epoch: int = 0
    last_lease_seq: int = -1
    last_counter_hi: int = -1


def verify_lease(
    vk: VerifyingKey,
    lease: Lease,
    *,
    expected_domain: bytes,
    now_ms: int,
    current_epoch: int,
    current_counter: int,
    verifier: "VerifierState | None" = None,
    enforce_seq: bool = True,
) -> tuple[bool, str]:
    """Executor-side check.

    If `verifier` is provided, the watermark is consulted/advanced atomically;
    `enforce_seq=False` mimics the NoSeq ablation (still verifies signatures
    / TTL / counter / epoch but does not require monotone lease_seq)."""
    if lease.domain != expected_domain:
        return False, "DOMAIN_MISMATCH"
    if not vk.verify(lease.message(), lease.signature):
        return False, "BAD_SIGNATURE"
    if now_ms > lease.exp_ms:
        return False, "EXPIRED"
    if lease.epoch != current_epoch:
        return False, "STALE_EPOCH"
    if not (lease.c_start <= current_counter <= lease.c_end):
        return False, "COUNTER_OUT_OF_WINDOW"
    if verifier is not None and enforce_seq:
        if lease.epoch < verifier.cur_epoch:
            return False, "STALE_EPOCH_HW"
        if lease.epoch == verifier.cur_epoch:
            if lease.lease_seq <= verifier.last_lease_seq:
                return False, "OUT_OF_ORDER"
        if lease.c_start <= verifier.last_counter_hi:
            return False, "COUNTER_NOT_MONOTONIC"
        verifier.cur_epoch = lease.epoch
        verifier.last_lease_seq = lease.lease_seq
        verifier.last_counter_hi = max(verifier.last_counter_hi, lease.c_end)
    return True, "OK"
