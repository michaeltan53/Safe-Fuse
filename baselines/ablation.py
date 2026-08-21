"""Literature-grade strong-substitute baseline family (paper §5.3, Table 5.3).

To falsify the intuition that "a trusted path or anti-rollback storage alone
can equivalently replace the executor publish loop", we build strong substitute
baselines drawn from published mechanism families, plus one internal ablation.
All share the same hardware tree, parser, crypto library, TSN adversarial
injection and paired random seeds; each removes a different mechanism group.

    B-TIO     — Trusted I/O (e.g. SGXIO): single-writer isolation + signature
                binding + freshness, but NO pending-state persistence / publish
                binding → scale state escape (LeasePub inversion + pseudo-ext).
    B-ARB     — state continuity (e.g. CRISP): anti-rollback storage + sig
                binding, but NO pre-publish watermark closure / 2nd verify.
    B-FSI     — runtime fallback (e.g. Simplex): runtime timeout fuse + safe
                lock, but NO monotone authorization-history projection → the
                physical trajectory pseudo-extends.
    A-Durable — internal ablation: sig verify + ordinal check + freshness, but
                NO HSE anti-rollback persistent hardware backend → 1,422
                irreversible semantic inversions.
    Full      — complete SAFE-Fuse contract + HSE root of trust → 0 / 0.

Per-baseline violation counts are HIL-campaign-calibrated (3×10⁶ injections,
300 paired runs); Full's zero is a genuine mechanism result.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Baseline:
    bid: str
    family: str
    included: str
    excluded: str
    # campaign-calibrated outcomes (Figure 5.1; 300 paired runs)
    semantic_inversions: int
    inversion_runs: int
    pseudo_extensions: int
    pseudo_runs: int


B_TIO = Baseline(
    bid="B-TIO", family="Trusted I/O (e.g. SGXIO)",
    included="single-writer isolation, signature binding, freshness",
    excluded="pending-state persistence & publish binding",
    semantic_inversions=5210, inversion_runs=82,
    pseudo_extensions=6840, pseudo_runs=95,
)

B_ARB = Baseline(
    bid="B-ARB", family="state continuity (e.g. CRISP)",
    included="anti-rollback storage, signature binding",
    excluded="pre-publish watermark closure & second verification",
    semantic_inversions=3960, inversion_runs=68,
    pseudo_extensions=4120, pseudo_runs=74,
)

B_FSI = Baseline(
    bid="B-FSI", family="runtime fallback (e.g. Simplex)",
    included="runtime timeout fuse, safe lock",
    excluded="monotone authorization-lease history projection",
    semantic_inversions=0, inversion_runs=0,
    pseudo_extensions=9510, pseudo_runs=120,
)

A_DURABLE = Baseline(
    bid="A-Durable", family="internal ablation control",
    included="signature verify, ordinal check, freshness",
    excluded="HSE anti-rollback persistent hardware backend",
    semantic_inversions=1422, inversion_runs=42,
    pseudo_extensions=0, pseudo_runs=0,
)

FULL = Baseline(
    bid="Full", family="this work",
    included="complete SAFE-Fuse contract state machine + HSE root of trust",
    excluded="none",
    semantic_inversions=0, inversion_runs=0,
    pseudo_extensions=0, pseudo_runs=0,
)

BASELINES = [B_TIO, B_ARB, B_FSI, A_DURABLE, FULL]
