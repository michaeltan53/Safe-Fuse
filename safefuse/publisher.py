"""Executor-side actuator verifier (AV) and the published trace `PubTrace`.

This module models the *executor pin* side of SAFE-Fuse — the object the new
evaluation chapter protects. A delivered lease only affects the physical
interface after the AV:

    1. verifies the signature / freshness / domain (cryptographic gate),
    2. checks the global high-water mark `w_c = (epoch, lease_seq)` so that
       only a strictly-monotone-increasing subsequence is ever published
       (defeats post-delivery reordering / stale-hold attacks),
    3. commits the new state to a double-buffered alternating-commit slot
       *before* publishing it to the pin (Commit-before-Publish), and
    4. publishes, appending a `PubEvent` to `PubTrace`.

A power cut between commit and publish is recovered deterministically via the
`pending_publish_fail` flag and pointer inversion, so the post-reboot pin
state never diverges from the last committed state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import List, Optional, Tuple

from .crypto import VerifyingKey
from .lease import Lease
from .state import AuthState


@dataclass
class PubEvent:
    """One row of PubTrace — a state that actually reached the pin."""

    t_publish_ms: float
    t_commit_ms: float
    epoch: int
    lease_seq: int
    state: AuthState
    is_active: bool        # ACCEPT or REJECT == active authorization


@dataclass
class PersistSlot:
    """One half of the double-buffered alternating-commit store."""

    epoch: int = 0
    lease_seq: int = -1
    state: AuthState = AuthState.FAILCLOSED


@dataclass
class ActuatorVerifier:
    """Local verifier running on the safety MCU (paper §5.1).

    Configurable knobs allow the ablation campaign (§5.4) to strip individual
    mechanisms:
        enforce_seq        — high-water (epoch, lease_seq) monotonicity
        commit_before_pub  — persist before publishing
        second_fresh_check — re-check TTL just before publishing (defeats
                             publish-after-expiry / scheduling hold)
        recovery_flag      — pending_publish_fail recovery on reboot
    """

    vk: VerifyingKey
    domain_id: bytes
    tau_rev_impl_ms: float
    delta_scan_ms: float = 0.2
    t_fb_ms: float = 1.2

    enforce_seq: bool = True
    commit_before_pub: bool = True
    second_fresh_check: bool = True
    recovery_flag: bool = True
    durable_wc: bool = True       # HSE anti-rollback persistence of w_c (§5.1)
    readback_verify: bool = True
    fault_latch: bool = True

    # high-water mark w_c
    cur_epoch: int = 0
    last_lease_seq: int = -1

    # double-buffered store + active pointer
    _slots: Tuple[PersistSlot, PersistSlot] = field(
        default_factory=lambda: (PersistSlot(), PersistSlot()))
    _active_slot: int = 0
    _pending_publish_fail: bool = False

    pub_trace: List[PubEvent] = field(default_factory=list)
    safe_trace: List[dict] = field(default_factory=list)
    physical_state: AuthState = AuthState.FAILCLOSED
    recovery_gate_active: bool = False
    durable_fault_locked: bool = False
    _fault_lock_write_pending: bool = False
    last_phase_ns: dict[str, int] = field(default_factory=dict)
    # bookkeeping for revocation experiments
    t_revoke_ms: Optional[float] = None

    # ------------------------------------------------------------------
    def _committed(self) -> PersistSlot:
        return self._slots[self._active_slot]

    def deliver(
        self,
        lease: Lease,
        *,
        now_ms: float,
        current_counter: int,
        readback_matches: bool = True,
    ) -> Tuple[bool, str]:
        """Process one delivered lease. Returns (published, reason)."""
        # ---- cryptographic / freshness gate ----
        # epoch/counter ordering is enforced by the high-water mark below, so
        # here we only require a valid signature, matching domain and TTL.
        verify_start = perf_counter_ns()
        sig_ok = self.vk.verify(lease.message(), lease.signature)
        if not sig_ok:
            return False, "BAD_SIGNATURE"
        if lease.domain != self.domain_id:
            return False, "DOMAIN_MISMATCH"
        if now_ms > lease.exp_ms:
            return False, "EXPIRED"

        # ---- high-water (epoch, lease_seq) monotonicity ----
        if self.enforce_seq:
            key_new = (lease.epoch, lease.lease_seq)
            key_hw = (self.cur_epoch, self.last_lease_seq)
            if key_new <= key_hw:
                return False, "BELOW_HIGH_WATER"     # blocks reorder/stale-hold
        verify_end = perf_counter_ns()

        # ---- commit-before-publish ----
        commit_start = perf_counter_ns()
        if self.commit_before_pub:
            t_commit = now_ms
            nxt = 1 - self._active_slot
            self._slots[nxt].epoch = lease.epoch
            self._slots[nxt].lease_seq = lease.lease_seq
            self._slots[nxt].state = lease.state
            self._pending_publish_fail = True
            # pointer inversion makes the new slot authoritative atomically
            self._active_slot = nxt
        else:
            # NoCommit ablation: publish with no preceding commit record →
            # PublishCommitGap. The sentinel falls outside the scan window.
            t_commit = -1.0e18
        commit_end = perf_counter_ns()

        # ---- second freshness check just before the pin write ----
        t_publish = now_ms + self.delta_scan_ms
        if self.second_fresh_check and t_publish > lease.exp_ms:
            self._pending_publish_fail = False
            return False, "PUBLISH_AFTER_EXPIRY"

        # ---- publish to the pin and independently read it back ----
        pin_start = perf_counter_ns()
        self.physical_state = lease.state
        if self.readback_verify and not readback_matches:
            self._pending_publish_fail = False
            if self.fault_latch:
                self.force_fail_closed(t_publish, reason="FEEDBACK_FAULT")
            self.last_phase_ns = {
                "verify": verify_end - verify_start,
                "durable_commit": commit_end - commit_start,
                "pin_readback": perf_counter_ns() - pin_start,
                "archive": 0,
            }
            return False, "READBACK_MISMATCH"

        self.cur_epoch = lease.epoch
        self.last_lease_seq = lease.lease_seq
        self._pending_publish_fail = False
        pin_end = perf_counter_ns()
        archive_start = perf_counter_ns()
        is_active = lease.state in (AuthState.ACCEPT, AuthState.REJECT)
        self.pub_trace.append(PubEvent(
            t_publish_ms=t_publish, t_commit_ms=t_commit,
            epoch=lease.epoch, lease_seq=lease.lease_seq,
            state=lease.state, is_active=is_active,
        ))
        self.last_phase_ns = {
            "verify": verify_end - verify_start,
            "durable_commit": commit_end - commit_start,
            "pin_readback": pin_end - pin_start,
            "archive": perf_counter_ns() - archive_start,
        }
        return True, "PUBLISHED"

    # ------------------------------------------------------------------
    def power_cut_and_reboot(self) -> bool:
        """Inject a hard power cut and reboot. Returns True if the recovered
        pin state diverges from the last committed state (a StateDivergence
        event). With the recovery flag + atomic double-buffer, divergence is
        impossible by construction."""
        if not self.recovery_flag:
            # no recovery flag: a pending publish leaves the pin in an
            # ambiguous half-written state → divergence.
            return self._pending_publish_fail
        # recovery flag set: re-derive the pin from the committed slot and
        # force fail-closed for any pending-but-unpublished commit.
        if self._pending_publish_fail:
            self._pending_publish_fail = False
            # Close the pin and write SafeTrace, not a synthetic LeasePub.
            now = self.pub_trace[-1].t_publish_ms if self.pub_trace else 0.0
            self.force_fail_closed(now, reason="POWER_RECOVERY")
        return False

    def rollback_watermark(self, to_epoch: int, to_seq: int) -> bool:
        """Attempt a power-loss-window rollback of the high-water mark to an
        older (epoch, seq) — the §5.5 weak-storage attack. Succeeds only when
        w_c is NOT on an anti-rollback (HSE Secure NVM) backend. Returns True
        if the rollback took effect (i.e. the durable guard is absent)."""
        if self.durable_wc:
            return False                 # HSE monotonic counter rejects rollback
        self.cur_epoch = to_epoch
        self.last_lease_seq = to_seq
        return True

    def force_fail_closed(self, now_ms: float, *, reason: str = "FAULT") -> None:
        """Drive the pin to fail-closed and record a non-LeasePub SafeTrace."""
        self.physical_state = AuthState.FAILCLOSED
        self.safe_trace.append({"t_safe_ms": now_ms + self.t_fb_ms, "reason": reason})

    def handle_clock_contract(self, now_ms: float, *, rtc_valid: bool,
                              rollback_detected: bool = False) -> Tuple[bool, str]:
        """Enforce the trusted-clock contract before ordinary publication.

        Returns ``(allowed, reason)``.  An invalid or rolled-back clock is a
        detectable contract failure and therefore latches the actuator into
        the fail-closed state rather than treating the time value as fresh.
        """
        if not rtc_valid:
            self.force_fail_closed(now_ms, reason="RTC_INVALID")
            return False, "RTC_INVALID"
        if rollback_detected:
            self.force_fail_closed(now_ms, reason="RTC_ROLLBACK")
            return False, "RTC_ROLLBACK"
        return True, "RTC_OK"

    def recover_from_storage_fault(
        self, now_ms: float, *, slot0_valid: bool, slot1_valid: bool,
        counter_consistent: bool = True,
    ) -> Tuple[bool, str]:
        """Recover one authenticated NVM slot or permanently fail closed.

        Validity represents the slot CRC/MAC and atomic-record checks.  If
        both slots are valid, the greatest ``(epoch, lease_seq)`` wins.  A
        counter inconsistency or absence of a valid snapshot is fail-closed.
        """
        if not counter_consistent:
            self.force_fail_closed(now_ms, reason="NVM_COUNTER_MISMATCH")
            return False, "NVM_COUNTER_MISMATCH"
        valid = [index for index, ok in enumerate((slot0_valid, slot1_valid)) if ok]
        if not valid:
            self.force_fail_closed(now_ms, reason="NVM_NO_VALID_SLOT")
            return False, "NVM_NO_VALID_SLOT"
        self._active_slot = max(valid, key=lambda index: (
            self._slots[index].epoch, self._slots[index].lease_seq))
        committed = self._committed()
        self.cur_epoch = committed.epoch
        self.last_lease_seq = committed.lease_seq
        self.physical_state = AuthState.FAILCLOSED
        self.safe_trace.append({"t_safe_ms": now_ms + self.t_fb_ms,
                                "reason": "NVM_SLOT_RECOVERY"})
        return True, "RECOVERED_VALID_SLOT"

    def begin_recovery(self) -> None:
        """Close the ordinary-admission gate before recovery work starts."""
        self.recovery_gate_active = True

    def finish_recovery(self) -> None:
        """Reopen admission only when no durable fault lock is active."""
        self.recovery_gate_active = self.durable_fault_locked

    def ordinary_admission_allowed(self) -> bool:
        """Return whether an ordinary Reserve/Drive may start."""
        return not self.recovery_gate_active and not self.durable_fault_locked

    def begin_durable_fault_lock_write(self) -> None:
        """Start the fail-safe durable fault-lock transition."""
        self.recovery_gate_active = True
        self._fault_lock_write_pending = True

    def power_cut_during_fault_lock_write(self, now_ms: float) -> None:
        """Model an interrupted lock record as locked on the next boot.

        This is the conservative encoding: an incomplete management record
        never re-enables ordinary admission.
        """
        if self._fault_lock_write_pending:
            self._fault_lock_write_pending = False
            self.durable_fault_locked = True
            self.recovery_gate_active = True
            self.force_fail_closed(now_ms, reason="DURABLE_FAULT_LOCK")
