"""
Ядро расчётов LTRROE: алгоритмы планирования и анализа рисков
Реализует прямой/обратный проходы, симуляцию Монте-Карло и корректировки на человеческий фактор
"""

from datetime import timedelta
import random
from typing import Dict, List, Tuple

def _iter_dependencies(project):
    """
    Возвращает зависимости независимо от того, хранятся ли они в виде списка или словаря.
    """
    dependencies = project.proj_dependencies
    if isinstance(dependencies, dict):
        return dependencies.values()
    return dependencies

def get_predecessors(project, task_id: int) -> List[int]:
    """
    Найти всех предшественников задачи
    """
    preds = []
    for dep in _iter_dependencies(project):
        if dep.dep_to_task == task_id:
            preds.append(dep.dep_from_task)
    return preds

def calculate_slowdown_factor(employee, task) -> float:
    """
    Рассчитать коэффициент замедления для сотрудника на задаче
    На основе несоответствия навыков и загрузки.
    Если эффективность выше 1, множитель становится меньше 1 и задача выполняется быстрее.
    """
    required_skills = task.task_skills or []
    employee_skills = employee.emp_skills or []

    # Проверяем недостающие навыки
    missing_skills = [skill for skill in required_skills if skill not in employee_skills]
    missing_count = len(missing_skills)
    total_count = len(required_skills)

    # Нет штрафа за навыки, если у задачи нет требуемых навыков
    if total_count == 0:
        skill_slowdown = 1.0
    # Отсутствуют все требуемые навыки
    elif missing_count == total_count:
        skill_slowdown = 3.0  # Очень медленно; плохое соответствие задаче

    # Отсутствуют некоторые требуемые навыки
    elif missing_count > 0:
        missing_ratio = missing_count / total_count
        base_penalty = 2.0

        # Базовый штраф 2.0 плюс дополнительный штраф за долю недостающих навыков
        additional_penalty = missing_ratio * 1.0
        skill_slowdown = base_penalty + additional_penalty
    else:
        # Находим минимальную эффективность среди требуемых навыков
        efficiencies = []
        for skill in required_skills:
            # Используем эффективность сотрудника по этому навыку, по умолчанию 0.20
            efficiency = (employee.emp_efficiency or {}).get(skill, 0.20)
            efficiencies.append(efficiency)

        min_efficiency = max(min(efficiencies), 0.01)

        # Коэффициент замедления по навыкам
        skill_slowdown = 1.0 / min_efficiency

    # Коэффициент замедления из-за перегрузки, если есть
    overload_slowdown = 1.0
    if employee.emp_current_load > employee.emp_max_daily_hours:
        overload = employee.emp_current_load - employee.emp_max_daily_hours
        # +5% за каждый избыточный час
        overload_slowdown = 1.0 + (overload * 0.05)

    # Итоговый коэффициент замедления
    total_slowdown = skill_slowdown * overload_slowdown
    
    return total_slowdown

def calculate_task_duration(task, project=None) -> float:
    """
    Рассчитать длительность задачи с учётом производительности исполнителя
    Использует формулу PERT для базовой оценки
    """
    # Базовая длительность: взвешенное среднее PERT
    base_duration = (task.task_duration_dist[0] + task.task_duration_dist[1] * 4 + task.task_duration_dist[2]) / 6

    # Возвращаем базовую длительность, если проект или назначение отсутствуют
    if project is None or not task.task_assigned_to:
        return base_duration

    # Безопасно получаем основного исполнителя
    try:
        primary_emp_id = task.task_assigned_to[0]
        employee = project.proj_employees.get(primary_emp_id)
        
        if employee is None:
            return base_duration
            
        slowdown = calculate_slowdown_factor(employee, task)
        return base_duration * slowdown
        
    except (IndexError, KeyError):
        return base_duration

def calculate_schedule(project) -> Tuple[Dict, Dict, Dict]:
    """
    Выполнить прямой проход для расчёта ранних дат начала и окончания
    Возвращает словари early_start, early_finish и task_duration
    """
    task_duration = {}  # task_id -> длительность в днях

    # Рассчитываем длительность для каждой задачи
    for task_id, task in project.proj_tasks.items():
        task_duration[task_id] = calculate_task_duration(task, project)
    
    early_start, early_finish = _forward_pass(project, task_duration)
    
    return early_start, early_finish, task_duration

def _forward_pass(project, task_duration: Dict) -> Tuple[Dict, Dict]:
    """
    Общий прямой проход для детерминированных и случайных длительностей задач.
    Возвращает словари early_start и early_finish
    """
    early_start = {}  # task_id -> дата начала
    early_finish = {}  # task_id -> дата окончания
    processed = set()

    while len(processed) < len(project.proj_tasks):
        progress = False

        for task_id in project.proj_tasks.keys():
            if task_id in processed:
                continue

            # Находим предшественников
            preds = get_predecessors(project, task_id)

            # Проверяем, можно ли обработать эту задачу
            if not preds or all(p in processed for p in preds):
                # Определяем дату начала
                if not preds:
                    # Нет зависимостей: начинаем с даты начала проекта
                    start_date = project.proj_start_date
                else:
                    # С зависимостями: начинаем после самого позднего окончания предшественника
                    max_finish_date = max(early_finish[p] for p in preds)
                    start_date = max_finish_date

                # Рассчитываем дату окончания
                duration_days = task_duration[task_id]
                finish_date = start_date + timedelta(days=duration_days)

                # Сохраняем результаты
                early_start[task_id] = start_date
                early_finish[task_id] = finish_date
                processed.add(task_id)
                progress = True

        if not progress:
            unresolved = sorted(set(project.proj_tasks) - processed)
            raise ValueError(
                "Cannot run the forward pass: check for cycles "
                f"or missing dependencies. Unresolved tasks: {unresolved}"
            )

    return early_start, early_finish

def get_successors(project, task_id: int) -> List[int]:
    """
    Найти всех последователей задачи
    """
    successors = []
    for dep in _iter_dependencies(project):
        if dep.dep_from_task == task_id:
            successors.append(dep.dep_to_task)
    return successors

def calculate_backward_pass(project, early_finish: Dict, task_duration: Dict) -> Tuple[Dict, Dict]:
    """
    Выполнить обратный проход для расчёта поздних дат начала и окончания
    Возвращает словари late_start и late_finish
    """
    late_start = {}
    late_finish = {}

    # Крайний срок проекта, без дополнительного буфера
    project_deadline = max(early_finish.values())

    # Инициализируем поздние даты окончания для конечных задач
    for task_id in project.proj_tasks.keys():
        succs = get_successors(project, task_id)
        if not succs:
            late_finish[task_id] = project_deadline

    # Обрабатываем задачи в порядке убывания early_finish
    tasks_sorted = sorted(project.proj_tasks.items(),
                         key=lambda x: early_finish[x[0]],
                         reverse=True)

    for task_id, task in tasks_sorted:
        succs = get_successors(project, task_id)

        if succs:
            # Находим минимальное позднее начало среди последователей
            min_late_start = min(late_start.get(s, project_deadline) for s in succs)
            late_finish[task_id] = min_late_start

        # Рассчитываем позднее начало
        late_start[task_id] = late_finish[task_id] - timedelta(days=task_duration[task_id])
    
    return late_start, late_finish

def random_triangular(low: float, most_likely: float, high: float) -> float:
    """
    Сгенерировать случайное значение из треугольного распределения
    Используется для симуляции PERT
    """
    if high == low:
        return low
    if high < low:
        raise ValueError(f"Invalid triangular distribution: high ({high}) < low ({low})")
    if not low <= most_likely <= high:
        raise ValueError(
            "Invalid triangular distribution: "
            f"most_likely ({most_likely}) must be between low ({low}) and high ({high})"
        )
    
    u = random.random()
    
    if u == 0:
        return low
    elif u == 1:
        return high
    
    # Нормализуем most_likely
    c = (most_likely - low) / (high - low)
    
    if u < c:
        return low + (u * (high - low) * (most_likely - low)) ** 0.5
    else:
        return high - ((1 - u) * (high - low) * (high - most_likely)) ** 0.5

def forward_pass_with_random_duration(project, random_duration: Dict) -> Dict:
    """
    Выполнить прямой проход со случайными длительностями задач
    Возвращает словарь early_finish для одной симуляции
    """
    _, early_finish = _forward_pass(project, random_duration)
    return early_finish

def build_task_slowdown_cache(project) -> Dict:
    """
    Предварительно вычислить замедление задачи с использованием основного исполнителя.
    Этот коэффициент остаётся неизменным внутри Монте-Карло, поэтому его не нужно
    пересчитывать в каждой симуляции.
    """
    task_slowdowns = {}

    for task_id, task in project.proj_tasks.items():
        slowdown = 1.0
        if task.task_assigned_to:
            primary_emp_id = task.task_assigned_to[0]
            employee = project.proj_employees.get(primary_emp_id)
            if employee:
                slowdown = calculate_slowdown_factor(employee, task)
        task_slowdowns[task_id] = slowdown

    return task_slowdowns

def monte_carlo_simulation(
    project,
    num_simulations: int = 1000,
    task_slowdowns: Dict = None
) -> List[float]:
    """
    Симуляция Монте-Карло для оценки рисков проекта
    Возвращает список длительностей проекта по всем симуляциям
    """
    project_durations = []
    if task_slowdowns is None:
        task_slowdowns = build_task_slowdown_cache(project)

    for sim in range(num_simulations):
        random_durations = {}

        for task_id, task in project.proj_tasks.items():
            # Генерируем базовую случайную длительность
            low, most_likely, high = task.task_duration_dist
            base_random = random_triangular(low, most_likely, high)

            # Корректируем с учётом производительности исполнителя
            adjusted_duration = base_random * task_slowdowns.get(task_id, 1.0)

            random_durations[task_id] = adjusted_duration

        # Выполняем прямой проход со случайными длительностями
        early_finish = forward_pass_with_random_duration(project, random_durations)
        
        if early_finish:
            max_finish_date = max(early_finish.values())
            project_duration = (max_finish_date - project.proj_start_date).days
            project_durations.append(project_duration)
    
    return project_durations
