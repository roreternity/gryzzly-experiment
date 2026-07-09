"""
Модели данных LTRROE
Определяет основные классы и структуры, используемые для представления сущностей проекта
Используется в исследовательском прототипе для типизированного, структурированного хранения данных

Основные классы:
- Task: задача проекта с длительностью, критичностью и требуемыми навыками
- Employee: член команды с навыками, загрузкой и атрибутами вероятности ошибки
- Project: контейнер, объединяющий задачи, сотрудников и зависимости
- Dependency: типизированная связь между задачами с временной задержкой
- Outsource: вариант внешнего выполнения задачи
- Assignment: назначение задачи сотруднику

Архитектура:
Классы содержат вспомогательные методы для типовых операций и производных параметров.
Каждый класс содержит атрибуты, необходимые для моделирования и анализа.
"""
 
from datetime import datetime
from typing import Dict, List, Optional, Union

EntityId = Union[int, str]

class Employee:
    def __init__(self, emp_id: EntityId, emp_name: str, emp_skills: List[str],
                 emp_error_prob: float, emp_cost_per_hour: float,
                 emp_efficiency: Dict[str, float]):
        self.emp_id = emp_id
        self.emp_name = emp_name
        self.emp_skills = emp_skills  # Список навыков сотрудника
        self.emp_error_prob = emp_error_prob
        self.emp_cost_per_hour = emp_cost_per_hour
        self.emp_efficiency = emp_efficiency  # Эффективность по навыку, где 0.6 = 60% и 1.2 = 120% от базового уровня
        self.emp_max_daily_hours = 8.0
        self.emp_current_load = 0.0
        self.emp_fatigue = 1.0  # Множитель усталости: >1 усталый, <1 отдохнувший; влияет на частоту ошибок и скорость
        self.emp_assigned_tasks = []  # Текущие назначенные задачи

class Task:
    def __init__(self, task_id: EntityId, task_name: str, task_skills: List[str], 
                 task_crit: int, task_cost: float, task_duration_dist: tuple):
        self.task_id = task_id
        self.task_name = task_name
        self.task_skills = task_skills  # Навыки, требуемые для задачи
        self.task_crit = task_crit  # Критичность задачи от 1 до 5, где 5 — наивысший приоритет
        self.task_cost = task_cost
        self.task_duration_dist = task_duration_dist
        self.task_assigned_to = []  # Назначенные сотрудники
        self.task_status = "in_progress"  # Текущий статус; допустимые значения: ['not_started', 'in_progress', 'completed', 'blocked']
        self.task_actual_duration = None
        self.task_primary_assignee = None

class Dependency:
    def __init__(self, dep_from_task: EntityId, dep_to_task: EntityId, 
                 dep_type: str, dep_lag: float, 
                 dep_mandatory: bool = True, dep_id: Optional[int] = None):
        self.dep_id = dep_id
        self.dep_from_task = dep_from_task  # Задача-предшественник
        self.dep_to_task = dep_to_task  # Задача-последователь
        self.dep_type = dep_type  # "FS", "SS", "FF", "SF"
        self.dep_lag = dep_lag  # Задержка в днях
        self.dep_mandatory = dep_mandatory

class Outsource:
    def __init__(self, outs_id: int, outs_name: str, outs_skills: List[str],
                 outs_daily_cost: float, outs_reliability: float,
                 outs_lead_time_days: int, outs_duration_multiplier: float = 1.5):
        self.outs_id = outs_id
        self.outs_name = outs_name
        self.outs_skills = outs_skills  # Навыки внешнего поставщика
        self.outs_daily_cost = outs_daily_cost
        self.outs_reliability = outs_reliability  # Общая надёжность/эффективность
        self.outs_lead_time_days = outs_lead_time_days  # Время выхода на работу (онбординг)
        self.outs_duration_multiplier = outs_duration_multiplier  # Множитель длительности задачи для внешнего поставщика (>1)

class Project:
    def __init__(self, proj_id=None):
        self.proj_id = proj_id
        self.proj_employees: Dict[EntityId, Employee] = {}  # Словарь сотрудников
        self.proj_tasks: Dict[EntityId, Task] = {}  # Словарь задач
        self.proj_dependencies: Dict[int, Dependency] = {}  # Словарь зависимостей
        self.proj_outsources: List[Outsource] = []  # Варианты внешнего выполнения
        self.proj_start_date = datetime.now()  # Дата начала проекта
        self.proj_current_date = datetime.now()  # Текущая дата симуляции для анализа "что если"
        self.proj_simulation_results = {}  # Хранилище результатов симуляции Монте-Карло
        self._next_dep_id = 1 # Счётчик ID зависимостей


    def add_dependency(self, dep_from_task: EntityId, dep_to_task: EntityId, 
                   dep_type: str, dep_lag: float, dep_mandatory: bool = True):
        dep = Dependency(
            dep_id=self._next_dep_id,
            dep_from_task=dep_from_task,
            dep_to_task=dep_to_task,
            dep_type=dep_type,
            dep_lag=dep_lag,
            dep_mandatory=dep_mandatory
        )
        self.proj_dependencies[self._next_dep_id] = dep
        self._next_dep_id += 1
        return dep
    
class Assignment:
    def __init__(self, asg_task_id: EntityId, asg_emp_id: EntityId, 
                 asg_planned_start: datetime, asg_planned_end: datetime,
                 asg_hours_per_day: float):
        self.asg_task_id = asg_task_id  # Назначенная задача
        self.asg_emp_id = asg_emp_id  # Назначенный сотрудник
        self.asg_planned_start = asg_planned_start  # Запланированная дата начала назначения
        self.asg_planned_end = asg_planned_end  # Запланированная дата окончания назначения
        self.asg_hours_per_day = asg_hours_per_day  # Дневная загрузка по этому назначению
        self.asg_actual_start = None
        self.asg_actual_end = None
        self.asg_progress = 0.0  # Прогресс выполнения назначения от 0.0 до 1.0
