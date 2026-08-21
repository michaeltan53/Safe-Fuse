"""Fully-defined audit chain with sentinel values (§3.4).

The chain head h_t is unconditionally updated on every microkernel step,
even when copy / binding / domain checks fail. Sentinel placeholders ⊥_r,
⊥_d, ⊥_c make the chain mapping-closed across all failure paths so that the
audit record stays verifiable end-to-end.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional

from .crypto import sha256
from .state import AuthState, ErrCode

SENTINEL_R = b"\x00\x00BOT_R\x00\x00"     # ⊥_r — copy failed
SENTINEL_D = b"\x00\x00BOT_D\x00\x00"     # ⊥_d — evidence invalid
SENTINEL_C = (1 << 64) - 1                # ⊥_c — counter unavailable


@dataclass
class AuditRecord:
    """One row of the audit chain. Kept in memory; production deployments
    would mirror this through the TEE secure log."""

    seq: int
    raw_hash: bytes      # H(raw_t^star) or ⊥_r
    state: AuthState
    digest: bytes        # d_t or H(raw*) or ⊥_d
    err: str
    epoch: int
    counter: int         # c_t or ⊥_c
    head: bytes          # h_t


class AuditChain:
    def __init__(self):
        self._head: bytes = sha256(b"safe-fuse:genesis")
        self._records: list[AuditRecord] = []

    @property
    def head(self) -> bytes:
        return self._head

    @property
    def records(self) -> list[AuditRecord]:
        return self._records

    def append(
        self,
        seq: int,
        raw_hash: Optional[bytes],
        state: AuthState,
        digest: Optional[bytes],
        err: ErrCode,
        epoch: int,
        counter: Optional[int],
    ) -> AuditRecord:
        r_bar = raw_hash if raw_hash is not None else SENTINEL_R
        d_bar = digest if digest is not None else SENTINEL_D
        c_bar = counter if counter is not None else SENTINEL_C

        # h_t = H( h_{t-1} || seq || \bar r_t || a_t || \bar d_t || err || g || \bar c )
        material = b"".join(
            [
                self._head,
                seq.to_bytes(8, "big"),
                r_bar,
                state.value.encode(),
                d_bar,
                err.value.encode(),
                epoch.to_bytes(4, "big"),
                struct.pack(">Q", c_bar),
            ]
        )
        new_head = sha256(material)
        rec = AuditRecord(
            seq=seq,
            raw_hash=r_bar,
            state=state,
            digest=d_bar,
            err=err.value,
            epoch=epoch,
            counter=c_bar,
            head=new_head,
        )
        self._records.append(rec)
        self._head = new_head
        return rec

    def verify(self) -> bool:
        """Independent contract auditor view (§3.1 'verify contract')."""
        head = sha256(b"safe-fuse:genesis")
        for rec in self._records:
            material = b"".join(
                [
                    head,
                    rec.seq.to_bytes(8, "big"),
                    rec.raw_hash,
                    rec.state.value.encode(),
                    rec.digest,
                    rec.err.encode(),
                    rec.epoch.to_bytes(4, "big"),
                    struct.pack(">Q", rec.counter),
                ]
            )
            head = sha256(material)
            if head != rec.head:
                return False
        return head == self._head
