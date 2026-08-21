"""Audit the raw HW-HIL inputs required by Figures 5.1, 5.2 and 5.3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.hil_evidence import audit_hil_inputs


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "hil_inputs")
    parser.add_argument("--report-only", action="store_true",
                        help="write the audit report without returning a failure status")
    args = parser.parse_args()
    report = audit_hil_inputs(args.input_dir)
    output = ROOT / "results" / "exp_5_11_hil_input_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"HW-HIL ready: {report['ready_for_hw_hil_figures']}")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    print(output)
    if not report["ready_for_hw_hil_figures"] and not args.report_only:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

