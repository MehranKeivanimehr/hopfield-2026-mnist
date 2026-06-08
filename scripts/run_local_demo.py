from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNS = [
    ("No CMKC ablation", "configs/ablation_no_cmkc.json", "artifacts/ablation_no_cmkc"),
    ("CMKC smoke", "configs/smoke.json", "artifacts/smoke"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the bundled CMKC local demo and write reports.")
    parser.add_argument("--skip-train", action="store_true", help="Only regenerate reports from existing artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for _, config_path, run_dir in RUNS:
        if not args.skip_train:
            run([sys.executable, "train.py", "--config", config_path])
        run([sys.executable, "scripts/report_results.py", "--run-dir", run_dir, "--config", config_path])

    report_path = Path("reports/local_demo.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_comparison(), encoding="utf-8")
    print(f"Wrote {report_path}")


def run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, check=True)


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def render_comparison() -> str:
    lines = [
        "# CMKC Local Demo Results",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "This report is produced by `scripts/run_local_demo.py` on the bundled toy continual-VQA sequence.",
        "It is a reproducibility check, not a claim of benchmark performance.",
        "",
        "| Run | Average accuracy | Average forgetting | Elapsed seconds |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, _, run_dir in RUNS:
        summary = read_json(Path(run_dir) / "summary.json")
        lines.append(
            "| {name} | {acc:.4f} | {forget:.4f} | {elapsed:.3f} |".format(
                name=name,
                acc=summary.get("average_accuracy", 0.0),
                forget=summary.get("average_forgetting", 0.0),
                elapsed=summary.get("elapsed_seconds", 0.0),
            )
        )

    lines.extend(
        [
            "",
            "## Included Runs",
            "",
            "- `configs/ablation_no_cmkc.json`: task loss only, no anchors, alignment distillation, or replay.",
            "- `configs/smoke.json`: full CMKC losses on the same tiny task sequence.",
            "",
            "## Next Benchmark Step",
            "",
            "Replace `data/sample_sequence.json` with a real continual VQA sequence, then repeat the same report workflow over at least three seeds.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
