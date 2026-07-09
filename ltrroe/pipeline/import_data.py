"""
Импортирует реальные CSV-данные Gryzzly в объекты LTRROE.
Строит проекты, сотрудников, задачи и зависимости.
"""

import pandas as pd
import random
import pickle
import re
from pathlib import Path
from collections import defaultdict
from ltrroe.core.objects import Project, Employee, Task

# Конфигурация
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "csvs"
FILES_DIR = BASE_DIR / "outputs"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

SKILL_POOL = [
    "Python", "Java", "JavaScript", "C++", "SQL", "DevOps",
    "ML", "UI/UX", "testing", "architecture", "databases", "documentation",
    "project management", "data analysis", "frontend", "backend"
]


_NS_PER_HOUR = 3_600_000_000_000  # наносекунды в часы

def parse_duration_str(x) -> float:
    """Разобрать строки вида '1h30m15s' в часы для tasks_computed."""
    if pd.isna(x):
        return 0.0
    if isinstance(x, (int, float)):
        # tasks_computed также может содержать число; трактуем его как часы
        return float(x)
    x = str(x).strip().lower()
    if x in ("0s", "0", ""):
        return 0.0
    h = re.search(r'(\d+)h', x)
    m = re.search(r'(\d+)m(?!s)', x)
    s = re.search(r'(\d+)s', x)
    return (int(h.group(1)) if h else 0) + \
           (int(m.group(1)) if m else 0) / 60 + \
           (int(s.group(1)) if s else 0) / 3600

def parse_duration_ns(x) -> float:
    """Преобразовать длительности деклараций из наносекунд в часы."""
    if pd.isna(x):
        return 0.0
    return float(x) / _NS_PER_HOUR

_NULL_STRINGS = {'', 'null', 'none', 'nan', 'nat', 'n/a', 'na'}

def is_null_str(val) -> bool:
    """Вернуть True для пустых значений и строковых представлений null."""
    if pd.isna(val):
        return True
    return str(val).strip().lower() in _NULL_STRINGS


def normalize_id(value):
    """
    Привести идентификаторы Gryzzly к строкам без потери совместимости.
    Родительские ID могут приходить как числа с `.0`; алгоритмы сравнивают ID напрямую.
    """
    if is_null_str(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def normalize_id_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Привести к строкам столбцы ID, присутствующие в датафрейме."""
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = df[column].apply(normalize_id)
    return df


# ── Загрузка CSV ──────────────────────────────────────────────────────────────
print("Loading CSV files...")
_NA = ['', 'null', 'NULL', 'None', 'NaN', 'nan', 'NA', 'N/A']
users             = pd.read_csv(DATA_DIR / "users.csv",             keep_default_na=True, na_values=_NA)
projects          = pd.read_csv(DATA_DIR / "projects.csv",          keep_default_na=True, na_values=_NA)
projects_computed = pd.read_csv(DATA_DIR / "projects_computed.csv", keep_default_na=True, na_values=_NA)
tasks             = pd.read_csv(DATA_DIR / "tasks.csv",             keep_default_na=True, na_values=_NA)
tasks_computed    = pd.read_csv(DATA_DIR / "tasks_computed.csv",    keep_default_na=True, na_values=_NA)
declarations      = pd.read_csv(DATA_DIR / "declarations.csv",      keep_default_na=True, na_values=_NA)

# Предобработка ─────────────────────────────────────────────────────────────
users = normalize_id_columns(users, ["id", "team_id"])
projects = normalize_id_columns(projects, ["id"])
projects_computed = normalize_id_columns(projects_computed, ["id"])
tasks = normalize_id_columns(tasks, ["id", "project_id", "parent_id"])
tasks_computed = normalize_id_columns(tasks_computed, ["id"])
declarations = normalize_id_columns(declarations, ["id", "user_id", "task_id"])

# Удалённые пользователи сохраняются, так как они участвовали в исторических задачах,
# а их декларации — реальные входные данные для прокси-показателя эффективности.
# Флаг is_active сохраняет это различие на случай, если оно понадобится.
users = users.copy()
users['is_active'] = users['deleted_at'].isna()
print(f"Total users: {len(users)} "
      f"(active: {users['is_active'].sum()}, "
      f"deleted: {(~users['is_active']).sum()})")

# Объединяем задачи и проекты с вычисляемыми таблицами
tasks_full    = tasks.merge(tasks_computed, on='id', how='left', suffixes=('', '_computed'))
projects_full = projects.merge(projects_computed, on='id', how='left', suffixes=('', '_computed'))

# Разбираем длительности задач/проектов из строк вида "1h30m"
for df in (tasks_full, projects_full):
    df['planned_duration'] = df['planned_duration'].apply(parse_duration_str)
    df['elapsed_duration'] = df['elapsed_duration'].apply(parse_duration_str)

# declarations.duration хранится в наносекундах, а не строкой длительности
declarations = declarations.copy()
declarations['duration'] = declarations['duration'].apply(parse_duration_ns)
print(f"Duration sample after parsing, hours: {declarations['duration'].head(3).tolist()}")

MIN_PROJECT_HOURS       = 1.0   # минимум 1 плановый час
MIN_TASK_HOURS          = 0.25  # минимум 15 минут для валидности PERT
MIN_PROJECT_DEPENDENCIES = 3    # проекты без зависимостей бесполезны для CPM-анализа
MAX_PROJECT_HOURS = 2000  # около 250 рабочих дней по 8 часов
MAX_PROJECT_PLANNED_ELAPSED_RATIO = 2.0

# Первичный фильтр проектов: плановая и фактическая длительности должны быть положительны
_candidate_projects = projects_full[
    (projects_full['planned_duration'] >= MIN_PROJECT_HOURS) &
    (projects_full['elapsed_duration'] >= MIN_PROJECT_HOURS) &
    (projects_full['planned_duration'] <= MAX_PROJECT_HOURS) &
    (projects_full['elapsed_duration'] <= MAX_PROJECT_HOURS) &
    (projects_full['planned_duration'] <= MAX_PROJECT_PLANNED_ELAPSED_RATIO * projects_full['elapsed_duration']) &
    (projects_full['elapsed_duration'] <= MAX_PROJECT_PLANNED_ELAPSED_RATIO * projects_full['planned_duration'])
]
_candidate_ids = set(_candidate_projects['id'])

# Группируем задачи по проекту один раз, чтобы не сканировать всю таблицу задач
# для каждого проекта заново.
_valid_tasks = tasks_full[
    tasks_full['project_id'].isin(_candidate_ids) &
    (tasks_full['planned_duration'] >= MIN_TASK_HOURS)
]
_tasks_by_proj  = {pid: grp for pid, grp in _valid_tasks.groupby('project_id')}

# parent_id → дочерние задачи для проверки зависимостей
_all_tasks_by_proj = {pid: grp for pid, grp in
                      tasks_full[tasks_full['project_id'].isin(_candidate_ids)].groupby('project_id')}

valid_project_ids = []

for proj_id in _candidate_ids:
    proj_tasks = _tasks_by_proj.get(proj_id)
    if proj_tasks is None or len(proj_tasks) < 2:
        continue

    if (proj_tasks['planned_duration'] <= 0).any():
        continue

    task_ids    = set(proj_tasks['id'])
    all_p_tasks = _all_tasks_by_proj.get(proj_id)
    if all_p_tasks is None:
        continue

    deps = all_p_tasks[
        all_p_tasks['parent_id'].isin(task_ids) &
        all_p_tasks['id'].isin(task_ids)
    ]
    if len(deps) < MIN_PROJECT_DEPENDENCIES:
        continue

    valid_project_ids.append(proj_id)

n_proj_dropped = len(projects_full) - len(valid_project_ids)
print(f"Projects passing filters: {len(valid_project_ids)} of {len(projects_full)} "
      f"(dropped: {n_proj_dropped})")

# Приводим valid_project_ids к Series для воспроизводимой выборки
valid_project_ids_s  = pd.Series(sorted(valid_project_ids))
sample_project_ids   = valid_project_ids_s.sample(min(5000, len(valid_project_ids_s)), random_state=RANDOM_SEED)
projects_full_sample = projects_full[projects_full['id'].isin(sample_project_ids)]

# В выборке только валидные задачи; MIN_TASK_HOURS уже гарантирован, но оставляем явный фильтр
_proj_tasks       = tasks_full[tasks_full['project_id'].isin(sample_project_ids)]
tasks_full_sample = _proj_tasks[_proj_tasks['planned_duration'] >= MIN_TASK_HOURS]
n_task_dropped    = len(_proj_tasks) - len(tasks_full_sample)
print(f"Tasks excluded (planned_duration<{MIN_TASK_HOURS}h): {n_task_dropped}")

declarations_sample = declarations[declarations['task_id'].isin(tasks_full_sample['id'])]

print(f"Sample: {len(projects_full_sample)} projects, "
      f"{len(tasks_full_sample)} tasks, "
      f"{len(declarations_sample)} declarations")


# ── Вспомогательные функции ───────────────────────────────────────────────────

def build_primary_map(declarations_df: pd.DataFrame) -> dict:
    """
    Строит карту {task_id: primary_user_id} по пользователю с наибольшим числом задекларированных часов.
    Вызывается один раз для всего датасета, а не внутри цикла по сотрудникам.
    """
    grouped = declarations_df.groupby(['task_id', 'user_id'])['duration'].sum().reset_index()
    idx = grouped.groupby('task_id')['duration'].idxmax()
    primary = grouped.loc[idx].set_index('task_id')['user_id'].to_dict()
    return primary


def get_employee_efficiency(user_id, tasks_df: pd.DataFrame,
                            primary_map: dict) -> float:
    """
    Прокси-эффективность = среднее отношение план/факт по задачам, где пользователь основной исполнитель.
    Принимает предвычисленный primary_map, чтобы не пересчитывать его внутри цикла.
    """
    user_task_ids = [tid for tid, uid in primary_map.items() if uid == user_id]
    if not user_task_ids:
        return 1.0
    subset = tasks_df[tasks_df['id'].isin(user_task_ids)]
    ratios = []
    for _, row in subset.iterrows():
        p, e = row['planned_duration'], row['elapsed_duration']
        if p > 0 and e > 0:
            ratios.append(p / e)
    if not ratios:
        return 1.0
    return max(0.5, min(2.0, sum(ratios) / len(ratios)))


def assign_skills_to_user(_user_id) -> list:
    return random.sample(SKILL_POOL, random.randint(2, 4))


# ── Предвычисляем primary_map один раз ──────────────────────────────────────────
primary_map = build_primary_map(declarations_sample)

# ── Строим объекты LTRROE ────────────────────────────────────────────────
# Итерируем только по отобранным проектам.
project_task_ids: dict[str, list[str]] = defaultdict(list)
for _, row in tasks_full_sample.iterrows():
    project_task_ids[row['project_id']].append(row['id'])

all_projects: dict = {}

print("Building projects...")
for proj_id, task_ids in project_task_ids.items():
    proj_info = projects_full_sample[projects_full_sample['id'] == proj_id]
    if proj_info.empty:
        continue

    proj_start      = pd.to_datetime(proj_info['created_at'].iloc[0])
    proj_planned_h  = proj_info['planned_duration'].iloc[0]
    proj_planned_days = max(proj_planned_h / 8.0, 1.0)  # запасной вариант >= 1 дня

    ltr_proj = Project(proj_id=proj_id)
    ltr_proj.proj_start_date = proj_start

    # Фильтруем декларации проекта один раз
    proj_declarations = declarations_sample[declarations_sample['task_id'].isin(task_ids)]

    # ── Сотрудники ────────────────────────────────────────────────────────────
    involved_user_ids = set(proj_declarations['user_id'].unique())
    employees_in_proj: dict = {}

    for uid in involved_user_ids:
        # Удалённые пользователи по-прежнему включаются, так как они реальные исторические участники.
        # user_row может быть пустым, если uid отсутствует в users.csv.
        user_row  = users[users['id'] == uid]
        is_active = bool(user_row['is_active'].iloc[0]) if not user_row.empty else False

        eff_value = get_employee_efficiency(uid, tasks_full_sample, primary_map)
        skills    = assign_skills_to_user(uid)
        eff_dict  = {skill: eff_value for skill in skills}

        emp = Employee(
            emp_id=uid,
            emp_name=f"User_{uid}",
            emp_skills=skills,
            emp_error_prob=0.1,
            emp_cost_per_hour=30.0,
            emp_efficiency=eff_dict
        )
        emp.emp_current_load = 0.0
        emp.emp_is_active = is_active   # True = сейчас активен, False = удалён/неактивен
        employees_in_proj[uid] = emp
        ltr_proj.proj_employees[uid] = emp

    # ── Задачи ────────────────────────────────────────────────────────────────
    for task_id in task_ids:
        rows = tasks_full_sample[tasks_full_sample['id'] == task_id]
        if rows.empty:
            continue
        task_row    = rows.iloc[0]
        planned_h   = task_row['planned_duration']
        elapsed_h   = task_row['elapsed_duration']

        # 1. Отбрасываем строки без полезной оценки длительности.
        if planned_h <= 0 and elapsed_h <= 0:
            continue

        # 2. Используем фактическую длительность как суррогат, если плановая оценка
        # отсутствует или слишком мала.
        base_h = max(planned_h, elapsed_h, 0.25)  # минимум 15 минут
        planned_days = base_h / 8.0

        # 3. Строим валидную тройку PERT: пессимистичная оценка должна строго превышать наиболее вероятную.
        a = max(0.25, planned_days * 0.7)
        m = max(a, planned_days)
        b = max(m + 0.01, planned_days * 1.5)

        if   m > 20: crit = 5
        elif m > 10: crit = 4
        elif m > 5:  crit = 3
        elif m > 2:  crit = 2
        else:        crit = 1

        ltr_task = Task(
            task_id=task_id,
            task_name=f"Task_{task_id}",
            task_skills=[],
            task_crit=crit,
            task_cost=0.0,
            task_duration_dist=(a, m, b)
        )

        # Основной исполнитель из предвычисленной карты
        primary_uid = primary_map.get(task_id)
        if primary_uid is not None and primary_uid in employees_in_proj:
            ltr_task.task_assigned_to.append(primary_uid)
            employees_in_proj[primary_uid].emp_assigned_tasks.append(task_id)

        ltr_proj.proj_tasks[task_id] = ltr_task

    # ── Зависимости (родитель → потомок) ─────────────────────────────────────────
    proj_tasks_df = tasks_full_sample[tasks_full_sample['project_id'] == proj_id]
    for _, row in proj_tasks_df.iterrows():
        parent = normalize_id(row.get('parent_id'))
        child  = normalize_id(row['id'])
        if parent and parent in ltr_proj.proj_tasks and child in ltr_proj.proj_tasks:
            ltr_proj.add_dependency(
                dep_from_task=parent,
                dep_to_task=child,
                dep_type="FS",
                dep_lag=0.0,
                dep_mandatory=True
            )

    # ── Текущая загрузка сотрудника ─────────────────────────────────────────
    for uid, emp in employees_in_proj.items():
        user_decl  = proj_declarations[proj_declarations['user_id'] == uid]
        total_hours = user_decl['duration'].sum()
        emp.emp_current_load = min(total_hours / proj_planned_days, 12.0)

    if len(ltr_proj.proj_dependencies) < MIN_PROJECT_DEPENDENCIES:
        continue

    all_projects[proj_id] = ltr_proj

    if len(all_projects) % 100 == 0:
        print(f"Processed projects: {len(all_projects)}")

print(f"Total projects built: {len(all_projects)}")

# Диагностика отобранных, но не построенных проектов
built_ids   = set(all_projects.keys())
sampled_ids = set(sample_project_ids)
lost_ids    = sampled_ids - built_ids
if lost_ids:
    print(f"\nSampled projects lost: {len(lost_ids)}")
    for pid in list(lost_ids)[:10]:
        proj_info = projects_full_sample[projects_full_sample['id'] == pid]
        n_tasks_raw   = len(tasks_full[tasks_full['project_id'] == pid])
        n_tasks_valid = len(tasks_full_sample[tasks_full_sample['project_id'] == pid])
        n_decl = len(declarations_sample[
            declarations_sample['task_id'].isin(
                tasks_full_sample[tasks_full_sample['project_id'] == pid]['id']
            )
        ])
        print(f"  {pid}: total tasks={n_tasks_raw}, "
              f"valid={n_tasks_valid}, declarations={n_decl}")

# ── Сохранение результата ────────────────────────────────────────────────────────────────
FILES_DIR.mkdir(parents=True, exist_ok=True)
output_file = FILES_DIR / "ltrroe_real_projects.pkl"
with open(output_file, "wb") as f:
    pickle.dump(all_projects, f)
print(f"Projects saved to {output_file}")

# ── Проверка результата ──────────────────────────────────────────────────────────────
if all_projects:
    sample_pid  = next(iter(all_projects))
    sample_proj = all_projects[sample_pid]
    print(f"\nSample project {sample_pid}:")
    print(f"  Employees:  {len(sample_proj.proj_employees)}")
    print(f"  Tasks:        {len(sample_proj.proj_tasks)}")
    print(f"  Dependencies: {len(sample_proj.proj_dependencies)}")
    print(f"  Start:       {sample_proj.proj_start_date}")
