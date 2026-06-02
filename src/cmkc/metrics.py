from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ContinualMetrics:
    task_accuracy_matrix: list[list[float]] = field(default_factory=list)

    def add_row(self, accuracies: list[float]) -> None:
        self.task_accuracy_matrix.append(accuracies)

    def average_accuracy(self) -> float:
        if not self.task_accuracy_matrix:
            return 0.0
        last_row = self.task_accuracy_matrix[-1]
        return sum(last_row) / len(last_row)

    def average_forgetting(self) -> float:
        if len(self.task_accuracy_matrix) < 2:
            return 0.0
        forgetting_values: list[float] = []
        num_tasks = len(self.task_accuracy_matrix[-1])
        for task_idx in range(num_tasks - 1):
            best = max(row[task_idx] for row in self.task_accuracy_matrix[:-1] if len(row) > task_idx)
            current = self.task_accuracy_matrix[-1][task_idx]
            forgetting_values.append(best - current)
        return sum(forgetting_values) / len(forgetting_values) if forgetting_values else 0.0
