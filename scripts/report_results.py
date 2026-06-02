from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Markdown report from CMKC run artifacts.")
    parser.add_argument("--run-dir", required=True, help="Directory containing summary.json.")
    parser.add_argument("--config", help="Optional config JSON used for the run.")
    parser.add_argument("--output", help="Output Markdown path. Defaults to <run-dir>/report.md.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    summary = read_json(run_dir / "summary.json")
    config = read_json(Path(args.config)) if args.config else None
    task_files = sorted(run_dir.glob("*_metrics.json"))
    task_metrics = [read_json(path) for path in task_files]

    output_path = Path(args.output) if args.output else run_dir / "report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_report(run_dir, summary, task_metrics, config), encoding="utf-8")
    print(f"Wrote {output_path}")


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def render_report(
    run_dir: Path,
    summary: dict[str, Any],
    task_metrics: list[dict[str, Any]],
    config: dict[str, Any] | None,
) -> str:
    lines = [
        "# CMKC Experiment Report",
        "",
        f"- Run directory: `{run_dir.as_posix()}`",
        f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- Average accuracy: {summary.get('average_accuracy', 0.0):.4f}",
        f"- Average forgetting: {summary.get('average_forgetting', 0.0):.4f}",
        "",
        "## Task Accuracy Matrix",
        "",
        "| After task | Evaluated task accuracies |",
        "| --- | --- |",
    ]

    matrix = summary.get("task_accuracy_matrix", [])
    for idx, row in enumerate(matrix, start=1):
        values = ", ".join(f"{value:.4f}" for value in row)
        lines.append(f"| {idx} | {values} |")

    if task_metrics:
        lines.extend(["", "## Per-Task Summaries", ""])
        lines.append("| Task | Seen tasks | Row accuracy | Average accuracy | Average forgetting |")
        lines.append("| --- | ---: | --- | ---: | ---: |")
        for item in task_metrics:
            row = ", ".join(f"{value:.4f}" for value in item.get("row_accuracy", []))
            lines.append(
                "| {task} | {seen} | {row} | {avg:.4f} | {forget:.4f} |".format(
                    task=item.get("task", "unknown"),
                    seen=item.get("seen_tasks", 0),
                    row=row,
                    avg=item.get("average_accuracy", 0.0),
                    forget=item.get("average_forgetting", 0.0),
                )
            )

    if config:
        selected_keys = [
            "seed",
            "device",
            "vision_backbone",
            "text_backbone",
            "hidden_dim",
            "epochs_per_task",
            "batch_size",
            "learning_rate",
            "lambda_vis_anchor",
            "lambda_lang_anchor",
            "lambda_align",
            "lambda_replay",
        ]
        lines.extend(["", "## Configuration", ""])
        lines.append("| Key | Value |")
        lines.append("| --- | --- |")
        for key in selected_keys:
            if key in config:
                lines.append(f"| `{key}` | `{config[key]}` |")

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- The smoke config is a reproducibility and wiring check, not a publishable benchmark.",
            "- For paper-grade numbers, replace the sample task sequence with VQACL or NExT-QA splits and report multiple seeds.",
            "- Compare CMKC against an ablation with replay/prototype/distillation weights set to zero.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
