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
class BenchmarkRow:
    method: str
    pixel_mode: str
    patterns_per_digit: int
    stored_patterns: int
    stored_pixels: int
    noise_type: str
    severity: float
    label_accuracy: float
    exact_retrieval_accuracy: float
    pixel_accuracy: float
    input_pixel_accuracy: float
    improvement: float
    mean_iterations: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2026 Hopfield MNIST associative memory lab.")
    parser.add_argument(
        "--mnist-dir",
        default=r"C:\Users\m.keivanimehr\OneDrive - University of Florida\Desktop\CV Project\Level 1",
        help="Directory containing train-images.idx3-ubyte and train-labels.idx1-ubyte.",
    )
    parser.add_argument("--output-dir", default="artifacts/hopfield_2026", help="Directory for results.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--trials-per-setting", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--variance-threshold", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = load_idx_images(Path(args.mnist_dir) / "train-images.idx3-ubyte")
    labels = load_idx_labels(Path(args.mnist_dir) / "train-labels.idx1-ubyte")

    rows: list[BenchmarkRow] = []
    capacity_grid = [1, 3, 5, 10]
    noise_grid = [
        ("bitflip", 0.20),
        ("saltpepper", 0.30),
        ("occlusion", 0.20),
    ]
    methods = ["nearest", "hebbian", "pseudoinverse", "modern_attention"]
    pixel_modes = ["full", "variance"]

    for patterns_per_digit in capacity_grid:
        stored_images, stored_labels = select_exemplars(images, labels, patterns_per_digit)
        full_patterns = binarize_bipolar(stored_images)
        for pixel_mode in pixel_modes:
            selected_pixels = select_pixels(full_patterns, pixel_mode, args.variance_threshold)
            patterns = full_patterns[:, selected_pixels]
            memories = build_memories(patterns)
            for noise_type, severity in noise_grid:
                queries = make_queries(
                    full_patterns,
                    selected_pixels,
                    noise_type,
                    severity,
                    args.trials_per_setting,
                    rng,
                )
                for method in methods:
                    rows.append(
                        evaluate_method(
                            method=method,
                            pixel_mode=pixel_mode,
                            patterns_per_digit=patterns_per_digit,
                            noise_type=noise_type,
                            severity=severity,
                            patterns=patterns,
                            labels=stored_labels,
                            queries=queries,
                            memories=memories,
                            max_iterations=args.max_iterations,
                            rng=rng,
                        )
                    )

    write_rows(output_dir / "benchmark.csv", rows)
    summary = aggregate_rows(rows)
    write_summary(output_dir / "summary_by_method.csv", summary)
    write_json(
        output_dir / "metadata.json",
        {
            "mnist_dir": str(Path(args.mnist_dir)),
            "seed": args.seed,
            "trials_per_setting": args.trials_per_setting,
            "max_iterations": args.max_iterations,
            "variance_threshold": args.variance_threshold,
            "num_train_images_available": int(images.shape[0]),
        },
    )
    plot_method_bar(summary, output_dir)
    plot_capacity_curves(rows, output_dir)
    plot_heatmap(rows, output_dir)
    plot_qualitative_gallery(images, labels, output_dir, args)
    render_report(output_dir, summary, rows)
    print(f"Wrote 2026 Hopfield lab results to {output_dir}")


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
    selected: list[int] = []
    for digit in range(10):
        matches = np.flatnonzero(labels == digit)
        selected.extend(matches[:per_digit].tolist())
    return images[selected], labels[selected]


def binarize_bipolar(images: np.ndarray) -> np.ndarray:
    flat = images.reshape(images.shape[0], -1)
    return np.where(flat > 127, 1, -1).astype(np.int8)


def select_pixels(patterns: np.ndarray, mode: str, variance_threshold: float) -> np.ndarray:
    if mode == "full":
        return np.arange(patterns.shape[1])
    mean_abs = np.abs(patterns.mean(axis=0))
    selected = np.flatnonzero(mean_abs < variance_threshold)
    if len(selected) == 0:
        raise ValueError("Variance filtering selected zero pixels.")
    return selected


def build_memories(patterns: np.ndarray) -> dict[str, np.ndarray]:
    x = patterns.astype(np.float32)
    hebbian = (x.T @ x) / x.shape[1]
    np.fill_diagonal(hebbian, 0.0)
    gram = x @ x.T
    pseudoinverse = x.T @ np.linalg.pinv(gram, rcond=1e-4) @ x
    np.fill_diagonal(pseudoinverse, 0.0)
    return {"hebbian": hebbian, "pseudoinverse": pseudoinverse}


def make_queries(
    full_patterns: np.ndarray,
    selected_pixels: np.ndarray,
    noise_type: str,
    severity: float,
    trials: int,
    rng: np.random.Generator,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    queries: list[tuple[int, np.ndarray, np.ndarray]] = []
    for index, clean_full in enumerate(full_patterns):
        for _ in range(trials):
            corrupted_full = corrupt(clean_full, noise_type, severity, rng)
            queries.append((index, clean_full[selected_pixels], corrupted_full[selected_pixels]))
    return queries


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


def evaluate_method(
    method: str,
    pixel_mode: str,
    patterns_per_digit: int,
    noise_type: str,
    severity: float,
    patterns: np.ndarray,
    labels: np.ndarray,
    queries: list[tuple[int, np.ndarray, np.ndarray]],
    memories: dict[str, np.ndarray],
    max_iterations: int,
    rng: np.random.Generator,
) -> BenchmarkRow:
    correct_label = 0
    exact = 0
    pixel_scores: list[float] = []
    input_scores: list[float] = []
    iterations: list[int] = []

    for clean_index, clean, query in queries:
        if method == "nearest":
            recalled = patterns[nearest_pattern(query, patterns)]
            iters = 0
        elif method == "modern_attention":
            recalled = modern_attention_recall(query, patterns)
            iters = 1
        elif method in {"hebbian", "pseudoinverse"}:
            recalled, iters = recurrent_recall(memories[method], query, rng, max_iterations)
        else:
            raise ValueError(f"Unknown method: {method}")

        predicted_index = nearest_pattern(recalled, patterns)
        correct_label += int(labels[predicted_index] == labels[clean_index])
        exact += int(predicted_index == clean_index)
        pixel_scores.append(pixel_accuracy(clean, recalled))
        input_scores.append(pixel_accuracy(clean, query))
        iterations.append(iters)

    n = len(queries)
    input_pixel = mean(input_scores)
    pixel = mean(pixel_scores)
    return BenchmarkRow(
        method=method,
        pixel_mode=pixel_mode,
        patterns_per_digit=patterns_per_digit,
        stored_patterns=int(patterns.shape[0]),
        stored_pixels=int(patterns.shape[1]),
        noise_type=noise_type,
        severity=severity,
        label_accuracy=correct_label / n,
        exact_retrieval_accuracy=exact / n,
        pixel_accuracy=pixel,
        input_pixel_accuracy=input_pixel,
        improvement=pixel - input_pixel,
        mean_iterations=mean(iterations),
    )


def recurrent_recall(
    weights: np.ndarray,
    query: np.ndarray,
    rng: np.random.Generator,
    max_iterations: int,
) -> tuple[np.ndarray, int]:
    state = query.astype(np.int8).copy()
    for iteration in range(1, max_iterations + 1):
        previous = state.copy()
        for idx in rng.permutation(state.shape[0]):
            state[idx] = 1 if weights[idx] @ state >= 0 else -1
        if np.array_equal(previous, state):
            return state, iteration
    return state, max_iterations


def modern_attention_recall(query: np.ndarray, patterns: np.ndarray, beta: float = 18.0) -> np.ndarray:
    x = patterns.astype(np.float32)
    q = query.astype(np.float32)
    scores = beta * (x @ q) / max(1.0, float(q.shape[0]))
    scores -= scores.max()
    weights = np.exp(scores)
    weights /= weights.sum()
    recalled = weights @ x
    return np.where(recalled >= 0, 1, -1).astype(np.int8)


def nearest_pattern(pattern: np.ndarray, patterns: np.ndarray) -> int:
    return int(np.argmin(np.mean(patterns != pattern[None, :], axis=1)))


def pixel_accuracy(clean: np.ndarray, other: np.ndarray) -> float:
    return float(np.mean(clean == other))


def mean(values: Iterable[float | int]) -> float:
    values = list(values)
    return float(sum(float(value) for value in values) / len(values))


def write_rows(path: Path, rows: list[BenchmarkRow]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].__dict__.keys()))
        writer.writeheader()
        writer.writerows(row.__dict__ for row in rows)


def aggregate_rows(rows: list[BenchmarkRow]) -> list[dict[str, float | str]]:
    groups: dict[tuple[str, str], list[BenchmarkRow]] = {}
    for row in rows:
        groups.setdefault((row.method, row.pixel_mode), []).append(row)
    summary: list[dict[str, float | str]] = []
    for (method, pixel_mode), items in sorted(groups.items()):
        summary.append(
            {
                "method": method,
                "pixel_mode": pixel_mode,
                "label_accuracy": mean(item.label_accuracy for item in items),
                "exact_retrieval_accuracy": mean(item.exact_retrieval_accuracy for item in items),
                "pixel_accuracy": mean(item.pixel_accuracy for item in items),
                "improvement": mean(item.improvement for item in items),
            }
        )
    return summary


def write_summary(path: Path, rows: list[dict[str, float | str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def plot_method_bar(summary: list[dict[str, float | str]], output_dir: Path) -> None:
    ordered = sorted(summary, key=lambda row: float(row["label_accuracy"]))
    labels = [pretty_method(str(row["method"])) + "\n" + pretty_pixel(str(row["pixel_mode"])) for row in ordered]
    label_acc = [float(row["label_accuracy"]) for row in ordered]
    exact_acc = [float(row["exact_retrieval_accuracy"]) for row in ordered]
    y = np.arange(len(labels))
    height = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.barh(y - height / 2, label_acc, height, label="Digit label retrieval", color="#2563eb")
    ax.barh(y + height / 2, exact_acc, height, label="Exact memory retrieval", color="#14b8a6")
    ax.set_xlim(0, 1.04)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Mean accuracy")
    ax.set_title("Associative Memory Retrieval on Corrupted MNIST", fontsize=15, pad=14)
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    for idx, value in enumerate(label_acc):
        ax.text(value + 0.015, idx - height / 2, f"{value:.2f}", va="center", fontsize=9)
    for idx, value in enumerate(exact_acc):
        ax.text(value + 0.015, idx + height / 2, f"{value:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "method_comparison.png", dpi=220)
    plt.close()


def plot_capacity_curves(rows: list[BenchmarkRow], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    colors = {
        "nearest": "#64748b",
        "hebbian": "#dc2626",
        "pseudoinverse": "#2563eb",
        "modern_attention": "#0f766e",
    }
    for method in ["nearest", "hebbian", "pseudoinverse", "modern_attention"]:
        for pixel_mode, style, alpha in [("full", "--", 0.78), ("variance", "-", 1.0)]:
            items = [row for row in rows if row.method == method and row.pixel_mode == pixel_mode]
            xs = sorted({row.stored_patterns for row in items})
            ys = [
                mean(row.label_accuracy for row in items if row.stored_patterns == stored)
                for stored in xs
            ]
            ax.plot(
                xs,
                ys,
                marker="o",
                linestyle=style,
                linewidth=2.4,
                alpha=alpha,
                color=colors[method],
                label=f"{pretty_method(method)} / {pretty_pixel(pixel_mode)}",
            )
    ax.set_xlabel("Number of stored MNIST memories")
    ax.set_ylabel("Mean digit-label retrieval accuracy")
    ax.set_title("Capacity Sweep Under Corrupted Queries", fontsize=15, pad=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, ncol=2, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(output_dir / "capacity_sweep.png", dpi=220)
    plt.close(fig)


def plot_heatmap(rows: list[BenchmarkRow], output_dir: Path) -> None:
    methods = ["hebbian", "modern_attention", "pseudoinverse", "nearest"]
    pixel_modes = ["full", "variance"]
    matrix = np.zeros((len(methods), len(pixel_modes)))
    for i, method in enumerate(methods):
        for j, pixel_mode in enumerate(pixel_modes):
            matrix[i, j] = mean(
                row.label_accuracy for row in rows if row.method == method and row.pixel_mode == pixel_mode
            )
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(np.arange(len(pixel_modes)), [pretty_pixel(mode) for mode in pixel_modes])
    ax.set_yticks(np.arange(len(methods)), [pretty_method(method) for method in methods])
    for i in range(len(methods)):
        for j in range(len(pixel_modes)):
            color = "white" if matrix[i, j] < 0.72 else "#111827"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=color, fontweight="bold", fontsize=13)
    ax.set_title("Mean Digit Retrieval Accuracy", fontsize=15, pad=14)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.set_label("Accuracy", rotation=270, labelpad=15)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_heatmap.png", dpi=220)
    plt.close(fig)


def plot_qualitative_gallery(
    images: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    rng = np.random.default_rng(args.seed + 99)
    stored_images, stored_labels = select_exemplars(images, labels, 1)
    full_patterns = binarize_bipolar(stored_images)
    var_pixels = select_pixels(full_patterns, "variance", args.variance_threshold)
    patterns = full_patterns[:, var_pixels]
    memories = build_memories(patterns)
    cases = [(2, "bitflip", 0.20), (5, "saltpepper", 0.30), (8, "occlusion", 0.20)]
    fig, axes = plt.subplots(len(cases), 5, figsize=(11.5, 6.8))
    for row, (digit, noise_type, severity) in enumerate(cases):
        idx = int(np.flatnonzero(stored_labels == digit)[0])
        clean = full_patterns[idx]
        corrupted = corrupt(clean, noise_type, severity, rng)
        query = corrupted[var_pixels]
        hebbian, _ = recurrent_recall(memories["hebbian"], query, rng, args.max_iterations)
        pseudo, _ = recurrent_recall(memories["pseudoinverse"], query, rng, args.max_iterations)
        modern = modern_attention_recall(query, patterns)
        panels = [
            ("Original", clean),
            (f"{noise_type}\nsev={severity}", corrupted),
            ("Hebbian", lift_to_full(hebbian, var_pixels)),
            ("Pseudoinverse", lift_to_full(pseudo, var_pixels)),
            ("Modern attention", lift_to_full(modern, var_pixels)),
        ]
        for col, (title, pattern) in enumerate(panels):
            axes[row, col].imshow((pattern.reshape(28, 28) + 1) / 2, cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title(title, fontsize=9.5)
            axes[row, col].axis("off")
    fig.suptitle("What Each Memory Recovers From the Same Corrupted Query", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "qualitative_gallery.png", dpi=220)
    plt.close(fig)


def lift_to_full(pattern: np.ndarray, selected_pixels: np.ndarray) -> np.ndarray:
    full = np.full(784, -1, dtype=np.int8)
    full[selected_pixels] = pattern
    return full


def pretty_method(method: str) -> str:
    return {
        "nearest": "Nearest neighbor",
        "hebbian": "Classical Hopfield",
        "pseudoinverse": "Pseudoinverse memory",
        "modern_attention": "Modern attention memory",
    }.get(method, method)


def pretty_pixel(pixel_mode: str) -> str:
    return {
        "full": "Full pixels",
        "variance": "Variance-filtered",
    }.get(pixel_mode, pixel_mode)


def render_report(output_dir: Path, summary: list[dict[str, float | str]], rows: list[BenchmarkRow]) -> None:
    best = max(summary, key=lambda row: float(row["label_accuracy"]))
    lines = [
        "# Hopfield 2026 MNIST Lab",
        "",
        "This run compares classical Hopfield recall with pseudoinverse memory and a modern Hopfield/attention-style retrieval rule on corrupted MNIST memories.",
        "",
        "References:",
        "",
        "- Krotov & Hopfield, Dense Associative Memory for Pattern Recognition, NeurIPS 2016.",
        "- Ramsauer et al., Hopfield Networks is All You Need, 2020.",
        "",
        "## Best Overall Configuration",
        "",
        f"- Method: `{best['method']}`",
        f"- Pixel mode: `{best['pixel_mode']}`",
        f"- Mean label retrieval accuracy: `{float(best['label_accuracy']):.4f}`",
        f"- Mean exact exemplar retrieval accuracy: `{float(best['exact_retrieval_accuracy']):.4f}`",
        "",
        "## Summary",
        "",
        "| Method | Pixel mode | Label acc | Exact retrieval | Pixel acc | Improvement |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| {method} | {pixel} | {label:.4f} | {exact:.4f} | {pixel_acc:.4f} | {improve:+.4f} |".format(
                method=row["method"],
                pixel=row["pixel_mode"],
                label=float(row["label_accuracy"]),
                exact=float(row["exact_retrieval_accuracy"]),
                pixel_acc=float(row["pixel_accuracy"]),
                improve=float(row["improvement"]),
            )
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `method_comparison.png`",
            "- `capacity_sweep.png`",
            "- `accuracy_heatmap.png`",
            "- `qualitative_gallery.png`",
            "",
            "## Scope",
            "",
            "This is still associative memory over stored MNIST examples, not general digit classification. The 2026 upgrade is the comparison of classical recurrence, pseudoinverse memory, and modern Hopfield/attention retrieval under the same corruption/capacity protocol.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
