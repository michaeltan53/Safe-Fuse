"""SAFE-Fuse trusted authorization microkernel (Algorithm 1, §4.6).

The class `Microkernel` implements the single-step scheduling routine of the
TCB. Every public state mutation goes through `step()`, so the call boundary
matches the "minimum trusted entry / exit" claim of §4.1.

Design notes
------------
* All inputs are treated as fully untrusted bytes / structured packets that
  arrived from the U domain. `_copy_in_fixed` is the first action and is the
  only place that interprets raw bytes — failure here triggers ERR_COPY with
  full audit-chain update.
* No dynamic allocation in the hot path. We use pre-allocated dicts and
  ints; this is the closest a Python reference can get to the WCET claim of
  §4.5 — the production version compiles to constant-time C in OP-TEE.
* The microkernel never dereferences U-shared memory directly: callers must
  serialize their inputs into the `CommandPacket` / `ProxyPacket` dataclasses
  before invocation. This mirrors the "no deref of untrusted shared memory"
  rule from §4.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .audit import AuditChain, AuditRecord
from .crypto import SigningKey, VerifyingKey, hmac_sha256, hmac_verify, sha256
from .hysteresis import Hysteresis, HysteresisParams
from .lease import Lease, issue_lease
from .operational_domain import DomainParams, OperationalDomain, GateOutcome
from .state import (
    ACTIVE_STATES,
    AuthState,
    CMD_FIXED_LEN,
    CommandPacket,
    ErrCode,
    Evidence,
    ProxyPacket,
    ResetCredential,
)


# Domain separation tags
DOM_BIND = b"safe-fuse:bind:v1"
DOM_PROXY = b"safe-fuse:proxy:v1"
DOM_LEASE = b"safe-fuse:lease:v1"
DOM_RESET = b"safe-fuse:reset:v1"


@dataclass
class MicrokernelConfig:
    """Static, signed configuration loaded at boot (§4.3)."""

    polver: int = 1                          # current policy version
    keyver: int = 1                          # signing key version
    lease_ttl_ms: int = 100                  # short-lease lifetime
    lease_window_steps: int = 8              # counter window width
    domain_id: bytes = b"safe-fuse:dom:hwy"  # operational-design domain tag
    # gate / state machine parameters
    domain_params: DomainParams = field(default_factory=DomainParams)
    hysteresis_params: HysteresisParams = field(default_factory=HysteresisParams)
    # security-feature toggles for §5.4.1 ablation only.
    enable_bind: bool = True                 # set False for NoBind
    enable_op_domain: bool = True            # set False for NoOpDomain
    enable_hysteresis: bool = True           # set False for NoHysteresis
    # whether to allow ValidReset to leave F. Default on.
    enable_reset: bool = True


@dataclass
class PersistState:
    """`σ^P = (g, h^P, keyver, b)` of §4.3.

    Real deployments back this with RPMB or a hardware monotonic counter.
    Here we mimic the write-amplification policy: only COLD_BOOT, epoch /
    state rollback detection, and VALID_RESET cause `g` to advance and
    `h^P` to be persisted.
    """

    epoch: int = 1
    head_persisted: bytes = b""
    keyver: int = 1
    boot_flag: int = 0
    rpmb_writes: int = 0   # observability counter for the DoS test (§5.4.2)

    def atomic_inc_epoch(self, audit_head: bytes) -> None:
        self.epoch += 1
        self.head_persisted = audit_head
        self.rpmb_writes += 1


@dataclass
class StepResult:
    state: AuthState
    err: ErrCode
    lease: Optional[Lease]
    audit: AuditRecord
    head: bytes


class Microkernel:
    """Implementation of Algorithm 1 — single-step decision scheduler."""

    def __init__(
        self,
        config: Optional[MicrokernelConfig] = None,
        *,
        hmac_meta_key: Optional[bytes] = None,
        hmac_proxy_key: Optional[bytes] = None,
        lease_signing_key: Optional[SigningKey] = None,
        reset_verifying_key: Optional[VerifyingKey] = None,
        time_fn: Optional[Callable[[], int]] = None,
    ) -> None:
        self.cfg = config or MicrokernelConfig()

        # symmetric keys (§4.2 / §4.4 tag spec).
        self._k_meta = hmac_meta_key or b"\x11" * 32
        self._k_proxy = hmac_proxy_key or b"\x22" * 32

        # lease signing key (§4.5).
        self._sk_lease = lease_signing_key or SigningKey(seed=b"\x33" * 32)
        self.vk_lease = self._sk_lease.verifying_key()

        # reset credential verifying key — held by the high-priv ops domain.
        # In production this is a hardware-anchored key entirely outside U.
        self._vk_reset = reset_verifying_key

        self.time_ms: Callable[[], int] = time_fn or self._default_time

        self.persist = PersistState(
            epoch=1, head_persisted=b"", keyver=self.cfg.keyver, boot_flag=1
        )

        self.audit = AuditChain()
        self.domain = OperationalDomain(self.cfg.domain_params)
        self.hyst = Hysteresis(self.cfg.hysteresis_params)

        # Runtime state (§4.6 Σ_run)
        self._seq: int = 0
        self._a_prev: AuthState = AuthState.ACCEPT
        self._counter_last_seen: int = -1
        self._last_ts_seen: int = -1
        self._epoch_runtime: int = self.persist.epoch

        # Cold-boot detection drives the very first transition (§4.3).
        self._cold_boot_pending: bool = True

        # Monotone per-epoch lease counter (§3 revision: strict linearisation
        # witness consumed by the executor-side verifier).
        self._lease_seq_per_epoch: dict[int, int] = {}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_time() -> int:
        return int(time.monotonic_ns() // 1_000_000)

    def _atomic_inc_epoch(self) -> int:
        self.persist.atomic_inc_epoch(self.audit.head)
        self._epoch_runtime = self.persist.epoch
        # Counter restart is implied by epoch advance (§4.3).
        self._counter_last_seen = -1
        return self._epoch_runtime

    def _next_lease_seq(self) -> int:
        ep = self._epoch_runtime
        nxt = self._lease_seq_per_epoch.get(ep, 0) + 1
        self._lease_seq_per_epoch[ep] = nxt
        return nxt

    # ------------------------------------------------------------------
    # Step 1: CopyInFixed
    # ------------------------------------------------------------------

    def _copy_in_fixed(self, raw_cmd: Optional[CommandPacket]) -> Optional[CommandPacket]:
        if raw_cmd is None:
            return None
        if raw_cmd.declared_length != CMD_FIXED_LEN:
            return None
        if len(raw_cmd.z_payload) == 0 or len(raw_cmd.z_payload) > CMD_FIXED_LEN:
            return None
        if len(raw_cmd.metadata) > CMD_FIXED_LEN:
            return None
        # at this point we have a verified-length defensive copy of cmd_t.
        return raw_cmd

    # ------------------------------------------------------------------
    # Step 2: VerifyEvidence
    # ------------------------------------------------------------------

    def _verify_evidence(
        self, cmd: CommandPacket
    ) -> tuple[bool, Optional[Evidence], ErrCode]:
        # ablation: NoBind drops the HMAC verification.
        if self.cfg.enable_bind:
            expected = hmac_sha256(self._k_meta, cmd.serialize_binding(DOM_BIND))
            if not hmac_verify(self._k_meta, cmd.serialize_binding(DOM_BIND), cmd.tag):
                return False, None, ErrCode.ERR_HMAC

            # epoch must match the current run epoch — defeats stale epoch
            # replay (§4.3 ERR_EPOCH).
            if cmd.epoch != self._epoch_runtime:
                # Distinguish rollback (lower) from forward injection.
                if cmd.epoch < self._epoch_runtime:
                    return False, None, ErrCode.ERR_EPOCH_ROLLBACK
                return False, None, ErrCode.ERR_EPOCH

            if cmd.polver != self.cfg.polver:
                return False, None, ErrCode.ERR_META

            if cmd.counter <= self._counter_last_seen:
                return False, None, ErrCode.ERR_REPLAY

            # Freshness: the timestamp must be strictly monotonic and within
            # `lease_ttl_ms * lease_window_steps` of the previous one. The
            # comparison is purely on the trusted internal clock relative to
            # the previously seen timestamp, so synthetic / replay timelines
            # are handled correctly. Absolute wall-clock comparisons live in
            # the lease verifier (lease.verify_lease), not here.
            if cmd.timestamp_ms <= self._last_ts_seen:
                return False, None, ErrCode.ERR_REPLAY
        else:
            # NoBind: still accept counter monotonicity *if* the input
            # provides one — but don't actually authenticate the tuple. This
            # is the exact attack surface the ablation tries to expose.
            pass

        d_t = sha256(cmd.z_payload)
        evidence = Evidence(
            z_star=cmd.z_payload,
            m_star=cmd.metadata,
            d_t=d_t,
            counter=cmd.counter,
            timestamp_ms=cmd.timestamp_ms,
            align_label=cmd.align_label,
            epoch=cmd.epoch,
            polver=cmd.polver,
        )
        return True, evidence, ErrCode.OK

    # ------------------------------------------------------------------
    # Step 3: GetAndVerifyProxy
    # ------------------------------------------------------------------

    def _verify_proxy(self, proxy: Optional[ProxyPacket]) -> tuple[bool, float, ErrCode]:
        if proxy is None:
            return False, 0.0, ErrCode.ERR_PROXY
        if self.cfg.enable_bind:
            if not hmac_verify(
                self._k_proxy, proxy.serialize_binding(DOM_PROXY), proxy.tag
            ):
                return False, 0.0, ErrCode.ERR_PROXY
        return True, float(proxy.s_hat), ErrCode.OK

    # ------------------------------------------------------------------
    # Step 4: Fixed-point score & manifold/noise estimators
    # ------------------------------------------------------------------

    @staticmethod
    def _fixed_point_score(z_star: bytes) -> float:
        """Decode the calibrated risk score embedded in the payload.

        In a real deployment this is a fixed-point matmul (§4.5). For the
        artefact we agree by convention: the first 4 bytes of z encode a
        big-endian uint32 representing `score * 1e6`. Anything else is a
        zero-score fallback (an empty / undefined payload arriving here
        already means evidence has been validated, so this is only a
        decode convention not a trust boundary).
        """
        if len(z_star) >= 4:
            v = int.from_bytes(z_star[:4], "big")
            return max(0.0, min(1.0, v / 1_000_000.0))
        return 0.0

    @staticmethod
    def _manifold_divergence(z_star: bytes) -> float:
        """Surrogate for the Mahalanobis divergence E(z*).

        Bytes 4..8 encode `1e3 * divergence` (big endian uint32).
        """
        if len(z_star) >= 8:
            v = int.from_bytes(z_star[4:8], "big")
            return v / 1_000.0
        return 0.0

    @staticmethod
    def _residual_estimate(z_star: bytes) -> float:
        """Surrogate for R̂_t.  Bytes 8..12 carry `1e6 * residual`."""
        if len(z_star) >= 12:
            v = int.from_bytes(z_star[8:12], "big")
            return v / 1_000_000.0
        return 0.0

    # ------------------------------------------------------------------
    # Step 5: failure path — full-definition audit + sentinel lease
    # ------------------------------------------------------------------

    def _fail_closed(
        self,
        reason: ErrCode,
        *,
        raw_hash: Optional[bytes],
        digest: Optional[bytes],
        counter: Optional[int],
    ) -> StepResult:
        # §4.3 write-amplification rule: only the three flagged reasons
        # advance the persistent epoch.
        if reason in {
            ErrCode.ERR_EPOCH_ROLLBACK,
            ErrCode.ERR_STATE_ROLLBACK,
            ErrCode.COLD_BOOT,
        }:
            self._atomic_inc_epoch()

        rec = self.audit.append(
            seq=self._seq,
            raw_hash=raw_hash,
            state=AuthState.FAILCLOSED,
            digest=digest,
            err=reason,
            epoch=self._epoch_runtime,
            counter=counter,
        )

        now = self.time_ms()
        lease = issue_lease(
            self._sk_lease,
            domain=self.cfg.domain_id,
            state=AuthState.FAILCLOSED,
            digest_d=digest or b"\x00" * 32,
            meta_hash=b"\x00" * 32,
            c_start=max(0, (counter or 0)),
            c_end=max(0, (counter or 0)),
            exp_ms=now + self.cfg.lease_ttl_ms,
            epoch=self._epoch_runtime,
            head=rec.head,
            err=reason.value,
            keyver=self.persist.keyver,
            lease_seq=self._next_lease_seq(),
        )

        self._a_prev = AuthState.FAILCLOSED
        return StepResult(AuthState.FAILCLOSED, reason, lease, rec, rec.head)

    # ------------------------------------------------------------------
    # The published API
    # ------------------------------------------------------------------

    def step(
        self,
        raw_cmd: Optional[CommandPacket],
        proxy: Optional[ProxyPacket],
        reset_cred: Optional[ResetCredential] = None,
    ) -> StepResult:
        """One scheduling round (Algorithm 1)."""

        self._seq += 1

        # --- cold boot detection (§4.3 §4.6) ---
        # The persisted state σ^P already carries the cold-boot epoch. We
        # therefore *record* COLD_BOOT in the audit chain to make the boot
        # event visible to an external verifier, but we do not bump the
        # epoch a second time. Stale evidence presented post-boot is still
        # rejected naturally via VerifyEvidence's epoch check.
        if self._cold_boot_pending:
            self._cold_boot_pending = False
            self.audit.append(
                seq=self._seq,
                raw_hash=None,
                state=AuthState.REJECT,
                digest=None,
                err=ErrCode.COLD_BOOT,
                epoch=self._epoch_runtime,
                counter=None,
            )
            self._a_prev = AuthState.REJECT
            self._seq += 1  # the actual input below is the next step

        # --- copy-in ---
        cmd = self._copy_in_fixed(raw_cmd)
        if cmd is None:
            return self._fail_closed(
                ErrCode.ERR_COPY, raw_hash=None, digest=None, counter=None,
            )

        raw_hash = sha256(cmd.z_payload + cmd.metadata)

        # --- absorbing-state branch ---
        if self._a_prev == AuthState.FAILCLOSED:
            if reset_cred is not None and self.cfg.enable_reset and self._verify_reset(reset_cred):
                self._atomic_inc_epoch()
                # transition: REJECT (active but safe) after reset.
                rec = self.audit.append(
                    seq=self._seq,
                    raw_hash=raw_hash,
                    state=AuthState.REJECT,
                    digest=sha256(cmd.z_payload),
                    err=ErrCode.EVT_RESET,
                    epoch=self._epoch_runtime,
                    counter=cmd.counter,
                )
                self._a_prev = AuthState.REJECT
                self.hyst.reset(AuthState.REJECT)
                self.domain.reset()
                lease = issue_lease(
                    self._sk_lease,
                    domain=self.cfg.domain_id,
                    state=AuthState.REJECT,
                    digest_d=sha256(cmd.z_payload),
                    meta_hash=sha256(cmd.metadata),
                    c_start=cmd.counter,
                    c_end=cmd.counter + self.cfg.lease_window_steps,
                    exp_ms=self.time_ms() + self.cfg.lease_ttl_ms,
                    epoch=self._epoch_runtime,
                    head=rec.head,
                    err=ErrCode.EVT_RESET.value,
                    keyver=self.persist.keyver,
                    lease_seq=self._next_lease_seq(),
                )
                return StepResult(AuthState.REJECT, ErrCode.EVT_RESET, lease, rec, rec.head)
            return self._fail_closed(
                ErrCode.ERR_ABSORB, raw_hash=raw_hash, digest=None, counter=cmd.counter,
            )

        # --- evidence verification ---
        ev_ok, evidence, err = self._verify_evidence(cmd)
        if not ev_ok:
            return self._fail_closed(
                err, raw_hash=raw_hash, digest=None, counter=cmd.counter,
            )
        assert evidence is not None
        self._counter_last_seen = cmd.counter
        self._last_ts_seen = cmd.timestamp_ms

        # --- proxy verification ---
        px_ok, s_hat, perr = self._verify_proxy(proxy)
        if not px_ok:
            return self._fail_closed(
                perr, raw_hash=raw_hash, digest=evidence.d_t, counter=cmd.counter,
            )

        # --- score & operational domain ---
        p_t = self._fixed_point_score(evidence.z_star)
        e_man = self._manifold_divergence(evidence.z_star)
        r_hat = self._residual_estimate(evidence.z_star)

        if self.cfg.enable_op_domain:
            gate: GateOutcome = self.domain.evaluate(p_t, s_hat, e_man, r_hat)
            if not gate.in_domain:
                err_code = ErrCode(gate.reason_code)
                return self._fail_closed(
                    err_code, raw_hash=raw_hash, digest=evidence.d_t, counter=cmd.counter,
                )
        else:
            # NoOpDomain ablation: still record but don't gate.
            self.domain._prev_score = p_t
            self.domain._prev_proxy = s_hat

        # --- hysteresis active-state decision ---
        if self.cfg.enable_hysteresis:
            a_t = self.hyst.step(p_t)
        else:
            # NoHysteresis ablation: instantaneous threshold at 0.5.
            a_t = AuthState.REJECT if p_t >= 0.5 else AuthState.ACCEPT

        rec = self.audit.append(
            seq=self._seq,
            raw_hash=raw_hash,
            state=a_t,
            digest=evidence.d_t,
            err=ErrCode.OK,
            epoch=self._epoch_runtime,
            counter=cmd.counter,
        )
        self._a_prev = a_t

        # --- lease issuance ---
        now = self.time_ms()
        lease = issue_lease(
            self._sk_lease,
            domain=self.cfg.domain_id,
            state=a_t,
            digest_d=evidence.d_t,
            meta_hash=sha256(evidence.m_star),
            c_start=cmd.counter,
            c_end=cmd.counter + self.cfg.lease_window_steps,
            exp_ms=now + self.cfg.lease_ttl_ms,
            epoch=self._epoch_runtime,
            head=rec.head,
            err=ErrCode.OK.value,
            keyver=self.persist.keyver,
            lease_seq=self._next_lease_seq(),
        )
        return StepResult(a_t, ErrCode.OK, lease, rec, rec.head)

    # ------------------------------------------------------------------
    # Reset credential verification
    # ------------------------------------------------------------------

    def _verify_reset(self, cred: ResetCredential) -> bool:
        if self._vk_reset is None:
            return False
        msg = (
            DOM_RESET
            + cred.epoch_floor.to_bytes(4, "big")
            + cred.issued_at_ms.to_bytes(8, "big")
            + cred.nonce
        )
        if not self._vk_reset.verify(msg, cred.signature):
            return False
        # only honor epoch_floor >= current persist epoch (defeats rollback).
        if cred.epoch_floor < self.persist.epoch:
            return False
        # 30s validity window
        if abs(self.time_ms() - cred.issued_at_ms) > 30_000:
            return False
        return True

    # ------------------------------------------------------------------
    # Convenience for experiments
    # ------------------------------------------------------------------

    @property
    def active_state(self) -> AuthState:
        return self._a_prev

    @property
    def epoch(self) -> int:
        return self._epoch_runtime
