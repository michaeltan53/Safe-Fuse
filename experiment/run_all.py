"""Run all Section-5 experiments from the reconstructed chapter."""
from __future__ import annotations

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

MODULES = [
    "experiments.exp_5_1_calibration",
    "experiments.exp_5_2_theorem_validation",
    "experiments.exp_5_3_external_baselines",
    "experiments.exp_5_4_ablation",
    "experiments.exp_5_4_gapcert",
    "experiments.exp_5_5_performance",
    "experiments.exp_5_6_sensitivity",
    "experiments.exp_5_7_additional",
    "experiments.exp_5_8_final_manuscript",
]


def main() -> None:
    for name in MODULES:
        print("=" * 78)
        print(name)
        print("=" * 78)
        importlib.import_module(name).main()
    from experiments.chapter5 import write_data_manifest
    write_data_manifest()


if __name__ == "__main__":
    main()
