"""Cryptographic primitives used by the SAFE-Fuse microkernel.

The paper standardizes three primitives (§4.5):
    * HMAC-SHA256 — metadata & proxy authentication
    * Ed25519     — short-lease and parameter-pack signing
    * SHA-256     — audit chain hashing

We use `cryptography` if available, falling back to `hashlib`+`hmac` for SHA
families. Ed25519 is provided through `cryptography`; if it is not installed
we fall back to a deterministic HMAC-based pseudo-signature that preserves the
unforgeability property under the paper's threat model (key is held entirely
inside the TCB process).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def hmac_sha256(key: bytes, data: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()


def hmac_verify(key: bytes, data: bytes, tag: bytes) -> bool:
    return hmac.compare_digest(hmac_sha256(key, data), tag)


# ----- Ed25519 wrappers -----------------------------------------------------

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.exceptions import InvalidSignature

    _HAVE_ED25519 = True
except Exception:  # pragma: no cover
    _HAVE_ED25519 = False


class SigningKey:
    """Opaque signing key. Lives entirely inside the TCB."""

    def __init__(self, seed: bytes | None = None):
        if _HAVE_ED25519:
            if seed is None:
                self._sk = Ed25519PrivateKey.generate()
            else:
                self._sk = Ed25519PrivateKey.from_private_bytes(seed[:32].ljust(32, b"\0"))
            self._pk = self._sk.public_key()
        else:
            self._sk = seed if seed is not None else secrets.token_bytes(32)
            self._pk = sha256(b"pk:" + self._sk)

    def sign(self, msg: bytes) -> bytes:
        if _HAVE_ED25519:
            return self._sk.sign(msg)
        # HMAC-SHA256 fallback signature.
        return hmac_sha256(self._sk, msg)

    def verifying_key(self) -> "VerifyingKey":
        return VerifyingKey(self._pk, fallback_secret=None if _HAVE_ED25519 else self._sk)


class VerifyingKey:
    def __init__(self, pk, fallback_secret: bytes | None = None):
        self._pk = pk
        self._fallback = fallback_secret  # only used when Ed25519 unavailable

    def verify(self, msg: bytes, sig: bytes) -> bool:
        if _HAVE_ED25519 and isinstance(self._pk, Ed25519PublicKey):
            try:
                self._pk.verify(sig, msg)
                return True
            except InvalidSignature:
                return False
            except Exception:
                return False
        # fallback path
        if self._fallback is None:
            return False
        return hmac.compare_digest(hmac_sha256(self._fallback, msg), sig)


def fresh_key(nbytes: int = 32) -> bytes:
    return secrets.token_bytes(nbytes)
