"""Authorization-stream state and fixed-length command/proxy packets.

Mirrors §3.1 and §4.2 of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# Fixed-length command packet size (bytes). The actual TA would pin this at
# compile time so that any oversize / undersize input is rejected by
# `CopyInFixed` before parsing — see §4.2.
CMD_FIXED_LEN = 512
PROXY_FIXED_LEN = 96


class AuthState(str, Enum):
    """Discrete authorization state space A = A_act ∪ {F}."""

    ACCEPT = "A"        # 𝖠 — pass through
    REJECT = "R"        # 𝖱 — soft reject
    FAILCLOSED = "F"    # 𝖥 — absorbing fail-closed


ACTIVE_STATES = {AuthState.ACCEPT, AuthState.REJECT}


@dataclass
class CommandPacket:
    """Untrusted command from U. Filled into a fixed-length buffer.

    The fields collectively form `raw_cmd_t` in Algorithm 1.
    """

    id_t: int                       # source id (sensor id)
    counter: int                    # monotonic counter c_t
    timestamp_ms: int               # τ_t
    align_label: int                # α_t — modality-alignment tag
    z_payload: bytes                # safety-relevant feature payload
    metadata: bytes                 # m_t (serialized)
    polver: int                     # policy version
    epoch: int                      # g_t (claimed)
    tag: bytes                      # HMAC over the binding tuple

    # the wire-level packet length the U layer pretends to be sending.
    # Anything != CMD_FIXED_LEN triggers ERR_COPY (§4.2).
    declared_length: int = CMD_FIXED_LEN

    def serialize_binding(self, domain: bytes) -> bytes:
        # tag = HMAC( dom || id || c || τ || α || d(z) || g || polver )
        # see §4.2 — the tag covers the full binding tuple including the
        # digest of z to close cross-modality misalignment attacks.
        import hashlib

        d = hashlib.sha256(self.z_payload).digest()
        parts = [
            domain,
            self.id_t.to_bytes(4, "big"),
            self.counter.to_bytes(8, "big"),
            self.timestamp_ms.to_bytes(8, "big"),
            self.align_label.to_bytes(4, "big"),
            d,
            self.epoch.to_bytes(4, "big"),
            self.polver.to_bytes(4, "big"),
        ]
        return b"".join(parts)


@dataclass
class ProxyPacket:
    """Authenticated physical proxy s_t (Table 4.2)."""

    id_source: int
    s_hat: float        # the normalized physical proxy value Ŝ_t
    counter: int
    timestamp_ms: int
    tag: bytes          # HMAC over (id, s, c, τ)

    def serialize_binding(self, domain: bytes) -> bytes:
        import struct

        return (
            domain
            + self.id_source.to_bytes(4, "big")
            + struct.pack(">d", float(self.s_hat))
            + self.counter.to_bytes(8, "big")
            + self.timestamp_ms.to_bytes(8, "big")
        )


@dataclass
class Evidence:
    """Parsed, validated evidence — populated only after VerifyEvidence."""

    z_star: bytes
    m_star: bytes
    d_t: bytes          # H(z*)
    counter: int
    timestamp_ms: int
    align_label: int
    epoch: int
    polver: int


@dataclass
class ResetCredential:
    """High-privilege ops credential used to leave F."""

    epoch_floor: int
    issued_at_ms: int
    nonce: bytes
    signature: bytes  # Ed25519 over the above fields


# Error codes used by FailClosed (matches the symbolic names in §4.6).
class ErrCode(str, Enum):
    OK = "OK"
    ERR_COPY = "ERR_COPY"
    ERR_META = "ERR_META"
    ERR_HMAC = "ERR_HMAC"
    ERR_REPLAY = "ERR_REPLAY"
    ERR_EPOCH = "ERR_EPOCH"
    ERR_EPOCH_ROLLBACK = "ERR_EPOCH_ROLLBACK"
    ERR_STATE_ROLLBACK = "ERR_STATE_ROLLBACK"
    ERR_PROXY = "ERR_PROXY"
    ERR_DOMAIN_SCORE = "ERR_DOMAIN_SCORE"
    ERR_DOMAIN_STEP = "ERR_DOMAIN_STEP"
    ERR_DOMAIN_MANIFOLD = "ERR_DOMAIN_MANIFOLD"
    ERR_DOMAIN_NOISE = "ERR_DOMAIN_NOISE"
    ERR_ABSORB = "ERR_ABSORB"
    COLD_BOOT = "COLD_BOOT"
    EVT_RESET = "EVT_RESET"
