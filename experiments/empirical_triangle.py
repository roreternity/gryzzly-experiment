"""Финальный пересчёт метрик с эмпирическим треугольным распределением.

Существующие pickle- и CSV-файлы не изменяются. Для каждой задачи текущее
наиболее вероятное значение сохраняется, а границы заменяются на 0,15 и 1,70
от него. CPM рассчитывается напрямую по наиболее вероятным плановым оценкам;
границы распределения используются только в Монте-Карло.
"""

from __future__ import annotations

import csv
import pickle
import random
import sys
from pathlib import Path
from statistics import mean, median


RANDOM_SEED = 42
NUM_SIMULATIONS = 10_000
MIN_TASKS = 4
LOW_FACTOR = 0.15
HIGH_FACTOR = 1.70

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = Path("/Users/roryqwork/Documents/Workspace/LTRROE_3/ltrroe_rus")
SOURCE_CODE = SOURCE_ROOT / "main"
SOURCE_PKL = SOURCE_ROOT / "files" / "ltrroe_real_projects.pkl"
OUTPUT_CSV = SCRIPT_DIR / "metrics_empirical_triangle_clean_10000.csv"

sys.path.insert(0, str(SOURCE_CODE))

import ltrroe_objects  # noqa: E402
from algorithms import (  # noqa: E402
    build_task_slowdown_cache,
    forward_pass_with_random_duration,
    monte_carlo_simulation,
)
from ltrroe_objects import Dependency  # noqa: E402


def percentile(sorted_values: list[float], q: float) -> float:
    index = min(len(sorted_values) - 1, int(len(sorted_values) * q))
    return sorted_values[index]


def normalize_dependencies(project) -> None:
    dependencies = project.proj_dependencies
    if isinstance(dependencies, dict):
        dependencies = list(dependencies.values())
    elif not isinstance(dependencies, list):
        dependencies = []
    project.proj_dependencies = [
        dependency
        for dependency in dependencies
        if isinstance(dependency, Dependency)
    ]


random.seed(RANDOM_SEED)
sys.modules.setdefault("models", ltrroe_objects)

with SOURCE_PKL.open("rb") as file:
    projects = pickle.load(file)

valid_projects = [
    project
    for project in projects.values()
    if len(project.proj_tasks) >= MIN_TASKS and project.proj_employees
]

rows = []
for index, project in enumerate(valid_projects, 1):
    normalize_dependencies(project)

    for task in project.proj_tasks.values():
        most_likely = float(task.task_duration_dist[1])
        task.task_duration_dist = (
            LOW_FACTOR * most_likely,
            most_likely,
            HIGH_FACTOR * most_likely,
        )

    task_slowdowns = build_task_slowdown_cache(project)
    planned_durations = {
        task_id: float(task.task_duration_dist[1]) * task_slowdowns[task_id]
        for task_id, task in project.proj_tasks.items()
    }
    early_finish = forward_pass_with_random_duration(project, planned_durations)
    deterministic_duration = (
        max(early_finish.values()) - project.proj_start_date
    ).days

    simulations = sorted(
        monte_carlo_simulation(
            project,
            num_simulations=NUM_SIMULATIONS,
            task_slowdowns=task_slowdowns,
        )
    )
    p50 = percentile(simulations, 0.50)
    p90 = percentile(simulations, 0.90)
    risk_ratio = (p90 - p50) / p50 if p50 > 0 else 0.0

    rows.append(
        {
            "project_id": getattr(project, "proj_id", f"project_{index}"),
            "n_tasks": len(project.proj_tasks),
            "det_duration_days": deterministic_duration,
            "p50": p50,
            "p90": p90,
            "schedule_risk_ratio": risk_ratio,
            "det_vs_p50_delta": p50 - deterministic_duration,
        }
    )

    if index % 100 == 0:
        print(f"Обработано: {index}/{len(valid_projects)}")

with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


def report(label: str, field: str) -> None:
    values = [row[field] for row in rows]
    print(
        f"{label:<34} "
        f"median={median(values):.3f}  mean={mean(values):.3f}"
    )


print(f"\nПроектов: {len(rows)}")
print(f"Распределение: Triangular({LOW_FACTOR}, 1.0, {HIGH_FACTOR})")
print(f"Симуляций на проект: {NUM_SIMULATIONS}")
report("Расчётный срок, дней", "det_duration_days")
report("P50, дней", "p50")
report("P90, дней", "p90")
report("Относительный резерв", "schedule_risk_ratio")
report("Разница P50 и расчётного срока", "det_vs_p50_delta")
print(f"CSV: {OUTPUT_CSV}")
