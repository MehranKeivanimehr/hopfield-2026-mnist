from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TrialResult:
    noise_type: str
    severity: float
    label: int
    trial: int
    input_accuracy: float
    recalled_accuracy: float
    improvement: float
    retrieval_correct: bool
    exact_match: bool
    iterations: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real Hopfield MNIST recovery experiment.")
    parser.add_argument(
        "--mnist-dir",
        default=r"C:\Users\m.keivanimehr\OneDrive - University of Florida\Desktop\CV Project\Level 1",
        help="Directory containing MNIST IDX files.",
    )
    parser.add_argument("--output-dir", default="artifacts/hopfield_mnist", help="Directory for CSVs and plots.")
    parser.add_argument("--patterns-per-digit", type=int, default=1, help="Stored exemplars per digit.")
    parser.add_argument("--trials", type=int, default=8, help="Random corruptions per stored pattern and severity.")
    parser.add_argument("--max-iterations", type=int, default=30, help="Synchronous recall iterations.")
    parser.add_argument(
        "--pixel-mode",
        choices=["full", "variance"],
        default="variance",
        help="Use all pixels or only pixels that vary across stored patterns.",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.8,
        help="For variance mode, keep pixels with abs(mean bipolar value) below this threshold.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = load_idx_images(Path(args.mnist_dir) / "train-images.idx3-ubyte")
    labels = load_idx_labels(Path(args.mnist_dir) / "train-labels.idx1-ubyte")
    stored_images, stored_labels = select_exemplars(images, labels, args.patterns_per_digit)
    full_patterns = binarize_bipolar(stored_images)
    selected_pixels = select_pixels(full_patterns, args.pixel_mode, args.variance_threshold)
    patterns = full_patterns[:, selected_pixels]
    weights = train_hopfield(patterns)

    severities = [0.05, 0.10, 0.20, 0.30, 0.40]
    noise_types = ["bitflip", "saltpepper", "occlusion"]
    results: list[TrialResult] = []

    for noise_type in noise_types:
        for severity in severities:
            for pattern_index, clean in enumerate(patterns):
                for trial in range(args.trials):
                    full_corrupted = corrupt(full_patterns[pattern_index], noise_type, severity, rng)
                    corrupted = full_corrupted[selected_pixels]
                    recalled, iterations = recall(weights, corrupted, rng, max_iterations=args.max_iterations)
                    input_acc = pixel_accuracy(clean, corrupted)
                    recalled_acc = pixel_accuracy(clean, recalled)
                    nearest = nearest_pattern(recalled, patterns)
                    results.append(
                        TrialResult(
                            noise_type=noise_type,
                            severity=severity,
                            label=int(stored_labels[pattern_index]),
                            trial=trial,
                            input_accuracy=input_acc,
                            recalled_accuracy=recalled_acc,
                            improvement=recalled_acc - input_acc,
                            retrieval_correct=nearest == pattern_index,
                            exact_match=bool(np.array_equal(clean, recalled)),
                            iterations=iterations,
                        )
                    )

    write_trials(output_dir / "trials.csv", results)
    summary = summarize(results)
    write_summary(output_dir / "summary.csv", summary)
    write_json(
        output_dir / "metadata.json",
        {
            "mnist_dir": str(Path(args.mnist_dir)),
            "num_train_images_available": int(images.shape[0]),
            "stored_patterns": int(patterns.shape[0]),
            "stored_pixels": int(patterns.shape[1]),
            "pixel_mode": args.pixel_mode,
            "variance_threshold": args.variance_threshold,
            "patterns_per_digit": args.patterns_per_digit,
            "trials_per_setting": args.trials,
            "max_iterations": args.max_iterations,
            "seed": args.seed,
        },
    )
    plot_summary(summary, output_dir)
    render_report(output_dir, summary)
    print(f"Wrote Hopfield MNIST results to {output_dir}")


def load_idx_images(path: Path) -> np.ndarray:
    with open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid IDX image file magic number {magic}: {path}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count, rows, cols)


def load_idx_labels(path: Path) -> np.ndarray:
    with open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid IDX label file magic number {magic}: {path}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count)


def select_exemplars(images: np.ndarray, labels: np.ndarray, per_digit: int) -> tuple[np.ndarray, np.ndarray]:
    selected_indices: list[int] = []
    for digit in range(10):
        matches = np.flatnonzero(labels == digit)
        if len(matches) < per_digit:
            raise ValueError(f"Need {per_digit} examples for digit {digit}, found {len(matches)}.")
        selected_indices.extend(matches[:per_digit].tolist())
    return images[selected_indices], labels[selected_indices]


def binarize_bipolar(images: np.ndarray) -> np.ndarray:
    flat = images.reshape(images.shape[0], -1)
    return np.where(flat > 127, 1, -1).astype(np.int8)


def select_pixels(patterns: np.ndarray, mode: str, variance_threshold: float) -> np.ndarray:
    if mode == "full":
        return np.arange(patterns.shape[1])
    if mode == "variance":
        mean_abs = np.abs(patterns.mean(axis=0))
        selected = np.flatnonzero(mean_abs < variance_threshold)
        if len(selected) == 0:
            raise ValueError("Variance pixel filter selected zero pixels.")
        return selected
    raise ValueError(f"Unknown pixel mode: {mode}")


def train_hopfield(patterns: np.ndarray) -> np.ndarray:
    weights = patterns.astype(np.float32).T @ patterns.astype(np.float32)
    np.fill_diagonal(weights, 0.0)
    weights /= patterns.shape[1]
    return weights


def corrupt(clean: np.ndarray, noise_type: str, severity: float, rng: np.random.Generator) -> np.ndarray:
    corrupted = clean.copy()
    if noise_type == "bitflip":
        mask = rng.random(clean.shape[0]) < severity
        corrupted[mask] *= -1
    elif noise_type == "saltpepper":
        mask = rng.random(clean.shape[0]) < severity
        corrupted[mask] = rng.choice(np.array([-1, 1], dtype=np.int8), size=int(mask.sum()))
    elif noise_type == "occlusion":
        image = corrupted.reshape(28, 28)
        side = max(1, int(round(math.sqrt(severity * 28 * 28))))
        top = int(rng.integers(0, 28 - side + 1))
        left = int(rng.integers(0, 28 - side + 1))
        image[top : top + side, left : left + side] = -1
        corrupted = image.reshape(-1)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    return corrupted


def recall(
    weights: np.ndarray,
    pattern: np.ndarray,
    rng: np.random.Generator,
    max_iterations: int,
) -> tuple[np.ndarray, int]:
    state = pattern.astype(np.int8).copy()
    num_neurons = state.shape[0]
    for iteration in range(1, max_iterations + 1):
        previous = state.copy()
        for idx in rng.permutation(num_neurons):
            state[idx] = 1 if weights[idx] @ state >= 0 else -1
        if np.array_equal(previous, state):
            return state, iteration
    return state, max_iterations


def pixel_accuracy(clean: np.ndarray, other: np.ndarray) -> float:
    return float(np.mean(clean == other))


def nearest_pattern(pattern: np.ndarray, stored_patterns: np.ndarray) -> int:
    distances = np.mean(stored_patterns != pattern[None, :], axis=1)
    return int(np.argmin(distances))


def write_trials(path: Path, results: Iterable[TrialResult]) -> None:
    rows = [result.__dict__ for result in results]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(results: list[TrialResult]) -> list[dict[str, float | str]]:
    groups: dict[tuple[str, float], list[TrialResult]] = {}
    for result in results:
        groups.setdefault((result.noise_type, result.severity), []).append(result)

    rows: list[dict[str, float | str]] = []
    for (noise_type, severity), items in sorted(groups.items()):
        rows.append(
            {
                "noise_type": noise_type,
                "severity": severity,
                "input_accuracy_mean": mean(item.input_accuracy for item in items),
                "recalled_accuracy_mean": mean(item.recalled_accuracy for item in items),
                "improvement_mean": mean(item.improvement for item in items),
                "retrieval_accuracy": mean(float(item.retrieval_correct) for item in items),
                "exact_match_rate": mean(float(item.exact_match) for item in items),
                "iterations_mean": mean(float(item.iterations) for item in items),
                "num_trials": float(len(items)),
            }
        )
    return rows


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values))


def write_summary(path: Path, rows: list[dict[str, float | str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def plot_summary(summary: list[dict[str, float | str]], output_dir: Path) -> None:
    for metric, ylabel in [
        ("recalled_accuracy_mean", "Mean recalled pixel accuracy"),
        ("retrieval_accuracy", "Stored-pattern retrieval accuracy"),
        ("improvement_mean", "Mean pixel-accuracy improvement"),
    ]:
        plt.figure(figsize=(7, 4.5))
        for noise_type in sorted({str(row["noise_type"]) for row in summary}):
            rows = [row for row in summary if row["noise_type"] == noise_type]
            xs = [float(row["severity"]) for row in rows]
            ys = [float(row[metric]) for row in rows]
            plt.plot(xs, ys, marker="o", linewidth=2, label=noise_type)
        plt.xlabel("Corruption severity")
        plt.ylabel(ylabel)
        plt.title(ylabel + " vs. corruption")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}.png", dpi=180)
        plt.close()


def render_report(output_dir: Path, summary: list[dict[str, float | str]]) -> None:
    lines = [
        "# Hopfield MNIST Recovery Results",
        "",
            "This experiment uses the real MNIST IDX files from the original CV Project folder.",
            "A Hopfield network stores one binarized exemplar per digit and recalls corrupted versions of those stored patterns.",
            "The recommended run uses variance-filtered pixels because full MNIST images are dominated by constant background pixels.",
        "",
        "| Noise | Severity | Input acc | Recalled acc | Improvement | Retrieval acc | Exact match | Iterations |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| {noise} | {severity:.2f} | {input_acc:.4f} | {recall_acc:.4f} | {improve:+.4f} | {retrieval:.4f} | {exact:.4f} | {iters:.2f} |".format(
                noise=row["noise_type"],
                severity=float(row["severity"]),
                input_acc=float(row["input_accuracy_mean"]),
                recall_acc=float(row["recalled_accuracy_mean"]),
                improve=float(row["improvement_mean"]),
                retrieval=float(row["retrieval_accuracy"]),
                exact=float(row["exact_match_rate"]),
                iters=float(row["iterations_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Plots",
            "",
            "- `recalled_accuracy_mean.png`",
            "- `retrieval_accuracy.png`",
            "- `improvement_mean.png`",
            "",
            "## Interpretation",
            "",
            "These are real local MNIST recovery results, but they evaluate associative recall of stored examples rather than general classification.",
            "The next GitHub step is to add a larger sweep over number of stored patterns, asynchronous recall, and comparison against denoising baselines.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
