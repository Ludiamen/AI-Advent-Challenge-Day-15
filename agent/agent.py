"""MemoryAgent — агент с явной моделью памяти.

Отличие этого дня от предыдущих в том, что у агента больше нет «истории» как
одной сущности. Есть три слоя с разным сроком жизни, разными хранилищами и
разными правилами записи, и каждый шаг агента проходит через них явно:

    вопрос
      -> маршрутизатор: нужно ли что-то сохранить надолго       (MemoryManager)
      -> запись реплики в краткосрочную память                  (MemoryManager)
      -> сборка промпта из слоёв по политике стадии             (PromptBuilder)
      -> вызов модели                                           (llm.Client)
      -> проверка ответа кодом по жёстким инвариантам           (StateValidator)
      -> при нарушении: повтор, затем эскалация модели
      -> запись ответа в краткосрочную память                   (MemoryManager)

Наружу агент отдаёт не только текст ответа, но и трейс: какие записи какого
слоя попали в промпт и во что это обошлось. Без трейса выполнить требование
задания «проверьте, какие данные попадают в каждый слой» нельзя — пришлось бы
верить на слово.

Публичное API:
  MemoryAgent(...)                  — создать агента
  .ask(question)                    — спросить с учётом всех включённых слоёв
  .plan()                           — составить план задачи и записать в рабочую память
  .start_task / .use_task / .transition / .finish_task
  .remember(target, ...)            — записать в указанный слой явно
  .set_layers(...)                  — включить и выключить слои (аблация)
  .info() / .stats() / .journal() / .files()
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from agent import catalog, interview, preferences, prompts, seed as seed_module
from agent.builder import BuiltPrompt, PromptBuilder
from agent.llm import Client, LLMError, Reply
from agent.memory.long import LongTermError
from agent.memory.manager import AUTO, LONG, SHORT, WORKING, MemoryManager
from agent.memory.router import DEFAULT_THRESHOLD, Routing
from agent.memory.short import DEFAULT_MAX_CHARS, DEFAULT_MAX_MESSAGES
from agent.memory.working import (
    DONE, PLANNING, ПО_КОМАНДЕ, ПРОДОЛЖИТЬ, TaskState, WorkingMemoryError,
)
from agent.preferences import Deviation, PreferenceChecker, SoftResult, StyleJudge
from agent.transitions import (
    КОД, МОДЕЛЬ, ЧЕЛОВЕК, ПереходОтклонён, Вердикт, Условие, Ворота,
    TransitionConfigError, просьба as просьба_о_переходе, убрать_маркер,
)
from agent.scenarios import RunResult, Scenario, ScenarioRunner
from agent.validator import Refusal, StateValidator, Violation, _разобрать_json

log = logging.getLogger("agent")

ALL_LAYERS = {SHORT, WORKING, LONG}

# Сколько раз агент пытается получить ответ, не нарушающий инварианты.
# Первая попытка — обычная; вторая — с напоминанием; третья — на модели
# следующей ступени. Дальше уже честнее отказать, чем жечь лимиты.
MAX_ATTEMPTS = 3

_ШАГ_ПЛАНА = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+(.{3,})$", re.MULTILINE)


class AgentError(RuntimeError):
    """Единственный тип ошибки, который агент выпускает наружу."""


@dataclass
class Answer:
    """Ответ агента вместе со всем, что понадобилось, чтобы его получить."""

    text: str
    prompt: BuiltPrompt | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    violations: list[Violation] = field(default_factory=list)
    deviations: list[Deviation] = field(default_factory=list)
    blocked: bool = False            # ответ так и не уложился в инварианты
    refusal: Refusal | None = None   # формализованный отказ вместо ответа
    применимые: list[str] = field(default_factory=list)   # коды учтённых инвариантов
    исправлено: list[str] = field(default_factory=list)   # нарушения, снятые повтором
    самоотчёт: list[str] = field(default_factory=list)    # что агент назвал сам
    вердикт: Any = None              # суждение проверяющей модели
    escalated_to: str = ""
    routing: Routing | None = None
    routing_entry: dict[str, Any] = field(default_factory=dict)
    soft: SoftResult | None = None
    # Попытка ассистента сменить стадию: что просил и чем кончилось.
    переход: dict[str, Any] = field(default_factory=dict)

    def layers(self) -> dict[str, dict[str, int]]:
        return self.prompt.by_layer() if self.prompt else {}

    def trace(self) -> list[dict[str, Any]]:
        return self.prompt.trace() if self.prompt else []

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "attempts": self.attempts,
            "blocked": self.blocked,
            "escalated_to": self.escalated_to,
            "usage": self.usage,
            "violations": [н.to_dict() for н in self.violations],
            "refusal": self.refusal.to_dict() if self.refusal else None,
            "применимые": self.применимые,
            "исправлено": self.исправлено,
            "самоотчёт": self.самоотчёт,
            "вердикт": self.вердикт.to_dict() if self.вердикт else None,
            "deviations": [о.to_dict() for о in self.deviations],
            "layers": self.layers(),
            "trace": self.trace(),
            "routing": self.routing.to_dict() if self.routing else None,
            "routing_entry": self.routing_entry,
            "soft": self.soft.to_dict() if self.soft else None,
            "переход": self.переход,
            "stage": self.prompt.stage if self.prompt else "",
            "task_id": self.prompt.task_id if self.prompt else "",
        }


class MemoryAgent:
    """Агент, у которого память разложена по трём слоям явно."""

    def __init__(
        self,
        model_key: str = "",
        user_id: str = "инженер",
        session: str = "основная",
        base_dir: str = "",
        task_id: str = "",
        layers: set[str] | None = None,
        router_mode: str = AUTO,
        threshold: float = DEFAULT_THRESHOLD,
        router_model: str = "",
        max_messages: int = DEFAULT_MAX_MESSAGES,
        max_chars: int = DEFAULT_MAX_CHARS,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        soft_check: bool = False,
        follow_profile: bool = True,
        judge_semantic: bool = True,
        require_self_report: bool = True,
        seed_project: bool = True,
    ) -> None:
        # Пустой model_key означает «брать модель по роли»: на планировании и
        # на исполнении роли разные, и жёстко фиксировать одну модель не нужно.
        self.model_key = model_key
        self.layers = set(layers) if layers is not None else set(ALL_LAYERS)
        self.soft_check = soft_check

        self.client = Client(temperature=temperature, max_tokens=max_tokens)
        база = base_dir or os.getenv("MEMORY_DIR", "memory")
        try:
            self.memory = MemoryManager(
                base_dir=база, user_id=user_id, session=session, client=self.client,
                router_mode=router_mode, threshold=threshold, router_model=router_model,
            )
        except (OSError, ValueError, LongTermError) as exc:
            # Сюда приходит и недопустимое имя пользователя: оно превращается в
            # путь, и проверка живёт в хранилище. Наружу это должно выйти
            # понятным отказом, а не стеком из недр памяти.
            raise AgentError(f"Не удалось открыть память: {exc}") from exc

        if seed_project:
            seed_module.seed(self.memory)

        # Ворота — единственное место, где меняется стадия задачи. Условия они
        # берут вызовом: пользователь может добавить своё условие посреди
        # работы, и следующий переход обязан его увидеть.
        self.ворота = Ворота(self.memory.all_conditions)
        self.builder = PromptBuilder(self.memory, max_messages=max_messages,
                                     max_chars=max_chars, gate=self.ворота)
        # Валидатор берёт инварианты вызовом, а не списком: их правят посреди
        # разговора, и слепок, снятый при создании агента, однажды окажется
        # вчерашним.
        self.validator = StateValidator(self.memory.all_invariants, self.client)
        self.style = StyleJudge(self.memory.long.profile, self.client)
        # Проверка предпочтений создаётся на каждый запрос заново: профиль
        # правят посреди разговора, и держать его слепок в поле значит однажды
        # проверить ответ по вчерашним настройкам.
        self.follow_profile = follow_profile
        # Два рубежа проверки смысловых инвариантов включаются порознь: первый
        # стоит один лишний вызов модели, второй — нескольких строк в ответе.
        self.judge_semantic = judge_semantic
        self.require_self_report = require_self_report
        self.runner = ScenarioRunner(self)

        self.task: TaskState | None = None
        if task_id:
            self.use_task(task_id)

    # --- свойства ------------------------------------------------------------

    @property
    def session(self) -> str:
        return self.memory.session

    @property
    def user_id(self) -> str:
        return self.memory.user_id

    @property
    def stage(self) -> str:
        return self.task.stage if self.task else ""

    def model_for(self, stage: str = "") -> str:
        """Какая модель отвечает на этой стадии.

        Планирование и исполнение разведены по ролям: на планировании цена
        ошибки выше, потому что на плане строится всё остальное.
        """
        if self.model_key:
            return self.model_key
        роль = "планирование" if (stage or self.stage) == PLANNING else "исполнение"
        return catalog.for_role(роль, offset=0)

    # --- слои ----------------------------------------------------------------

    def set_layers(self, layers: set[str]) -> None:
        """Включает и выключает слои. Этим делается аблация в compare.py."""
        неизвестные = layers - ALL_LAYERS
        if неизвестные:
            raise AgentError(
                f"Неизвестные слои: {', '.join(неизвестные)}. "
                f"Допустимы: {', '.join(sorted(ALL_LAYERS))}."
            )
        self.layers = set(layers)

    # --- задачи --------------------------------------------------------------

    def start_task(self, task_id: str, title: str = "", overwrite: bool = False) -> TaskState:
        """Заводит задачу и делает её текущей."""
        try:
            self.task = self.memory.working.create(task_id, title, overwrite=overwrite)
        except WorkingMemoryError as exc:
            raise AgentError(str(exc)) from exc
        self.memory._log("шаг-задачи", WORKING, task_id, title, applied=True,
                         reason="задача заведена, стадия planning")
        return self.task

    def use_task(self, task_id: str) -> TaskState:
        """Поднимает задачу из рабочей памяти — в том числе спустя дни."""
        try:
            self.task = self.memory.working.load(task_id)
        except WorkingMemoryError as exc:
            raise AgentError(str(exc)) from exc
        return self.task

    def drop_task(self) -> None:
        self.task = None

    def save_task(self) -> None:
        """Кладёт состояние текущей задачи на диск.

        Вызывается после каждого шага и каждой остановки: именно записанное
        состояние, а не переменная в памяти процесса, позволяет вернуться к
        задаче завтра и с другого запуска.
        """
        if self.task is not None:
            self.memory.working.save(self.task)

    # --- пауза и продолжение -------------------------------------------------

    def pause_task(self, причина: str = ПО_КОМАНДЕ, пояснение: str = "") -> TaskState:
        """Просит задачу остановиться. Текущий шаг при этом доводится до конца.

        Флаг ставится на том же объекте состояния, с которым работает исполнитель
        сценария, поэтому он увидит паузу перед следующим шагом. Прерывать шаг
        посреди вызова модели незачем: ответ уже оплачен, и выбрасывать его —
        значит платить дважды.
        """
        if self.task is None:
            raise AgentError("Активной задачи нет — останавливать нечего.")
        if self.task.finished:
            raise AgentError("Задача уже завершена.")
        self.task.остановить(причина, ПРОДОЛЖИТЬ, пояснение)
        self.save_task()
        self.memory._log("шаг-задачи", WORKING, self.task.task_id,
                         f"пауза: {причина}", applied=True,
                         reason=пояснение or "остановлено по команде")
        return self.task

    def continue_task(self) -> TaskState:
        """Снимает паузу, не запуская исполнение. Само продолжение — у сценария."""
        if self.task is None:
            raise AgentError("Активной задачи нет.")
        self.task.продолжить()
        self.save_task()
        return self.task

    def answer_task(self, текст: str) -> TaskState:
        """Принимает ответ человека на остановку «не хватает сведений»."""
        if self.task is None:
            raise AgentError("Активной задачи нет.")
        try:
            self.task.ответить(текст)
        except WorkingMemoryError as exc:
            raise AgentError(str(exc)) from exc
        self.save_task()
        self.memory._log("шаг-задачи", WORKING, self.task.task_id,
                         текст, applied=True, reason="ответ человека на вопрос шага")
        return self.task

    def resume_scenario(
        self,
        task_id: str = "",
        ответ: str = "",
        on_step: Any = None,
        on_result: Any = None,
        по_шагам: bool = False,
    ) -> RunResult:
        """Продолжает отложенную задачу с того шага, где она стоит."""
        task_id = task_id or (self.task.task_id if self.task else "")
        if not task_id:
            raise AgentError("Не указано, какую задачу продолжать.")
        try:
            return self.runner.resume(task_id, ответ=ответ, on_step=on_step,
                                      on_result=on_result, по_шагам=по_шагам)
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError(f"Не удалось продолжить задачу «{task_id}»: {exc}") from exc

    def task_state(self) -> dict[str, Any]:
        """Полное состояние задачи для интерфейсов: этап, шаг, ожидание, пауза."""
        if self.task is None:
            return {"есть": False}
        з = self.task
        return {
            "есть": True,
            "task_id": з.task_id,
            "title": з.title,
            "сценарий": з.сценарий,
            "этап": з.stage,
            "этап_словами": з.stage_label,
            "разрешено": list(з.allowed()),
            "шаг": з.текущий_шаг,
            "шагов": з.шагов,
            "шаг_словами": з.шаг_словами,
            "шаги": [ш.to_dict() for ш in з.шаги],
            "ожидание": з.ожидание,
            "ожидание_текст": з.ожидание_текст,
            "пауза": з.пауза,
            "причина_паузы": з.причина_паузы,
            "ответы": list(з.ответы),
            "собрано": len(з.collected),
            "словами": з.состояние_словами,
            # --- ворота: чем открывается следующий этап ----------------------
            "план": list(з.plan),
            "план_утверждён": dict(з.план_утверждён),
            "утверждение_актуально": з.утверждение_актуально,
            "отчёт_валидации": dict(з.отчёт_валидации),
            "валидация_актуальна": з.валидация_актуальна,
            "переходы": self.ворота.обзор(з),
            "история_переходов": list(з.transitions),
            "отказы": list(з.отказы),
        }

    def transition(self, stage: str, note: str = "", кто: str = ЧЕЛОВЕК) -> TaskState:
        """Переводит задачу на новую стадию — если выполнены условия перехода.

        Проверок две, и они разной природы: есть ли такая стрелка в жизненном
        цикле и заслужен ли шаг. Вторую делает Ворота; отклонённая попытка
        ложится в журнал задачи вместе с тем, кто её предпринял.
        """
        if self.task is None:
            raise AgentError("Активной задачи нет: сначала --задача или --новая-задача.")
        вердикт = self.ворота.перевести(self.task, stage, кто, note)
        self.memory.working.save(self.task)
        if not вердикт.можно:
            self.memory._log("шаг-задачи", WORKING, self.task.task_id,
                             f"-X-> {stage}", applied=False,
                             reason=f"переход отклонён ({кто}): {вердикт.причина}")
            raise ПереходОтклонён(вердикт.отказ)
        self.memory._log("шаг-задачи", WORKING, self.task.task_id, f"-> {stage}",
                         applied=True, reason=note or f"смена стадии ({кто})")
        return self.task

    def check_transition(self, stage: str, кто: str = ЧЕЛОВЕК) -> Вердикт:
        """Вердикт по переходу, ничего не меняя: этим живут кнопки интерфейсов."""
        if self.task is None:
            raise AgentError("Активной задачи нет.")
        return self.ворота.проверить(self.task, stage, кто)

    def transition_options(self) -> list[dict[str, Any]]:
        """Все стадии с пометкой «можно/нельзя и почему» — для страницы и CLI."""
        if self.task is None:
            return []
        return self.ворота.обзор(self.task)

    # --- ворота: подпись под планом и отчёт проверки -------------------------

    def approve_plan(self, кем: str = "", пояснение: str = "") -> dict[str, Any]:
        """Утверждает план задачи — этим открывается переход к реализации."""
        if self.task is None:
            raise AgentError("Активной задачи нет: утверждать нечего.")
        try:
            подпись = self.task.утвердить_план(кем or self.user_id, пояснение)
        except WorkingMemoryError as exc:
            raise AgentError(str(exc)) from exc
        self.memory.working.save(self.task)
        self.memory._log("шаг-задачи", WORKING, self.task.task_id,
                         f"план утверждён ({подпись['пунктов']} п.)", applied=True,
                         reason=f"подпись под планом: {подпись['кем']}")
        return подпись

    def unapprove_plan(self, почему: str = "") -> None:
        """Снимает подпись с плана: план решили переделать."""
        if self.task is None:
            raise AgentError("Активной задачи нет.")
        self.task.снять_утверждение(почему)
        self.memory.working.save(self.task)

    def validate_task(self, ревизор: bool = True, model_key: str = "") -> dict[str, Any]:
        """Собирает отчёт проверки — этим открывается переход в done.

        Чек-лист собирает код: он смотрит на состояние задачи, а не на мнение о
        нём. Модель-ревизор добавляет к нему смысловой вердикт — то, чего код не
        видит: сходится ли сделанное с планом по существу. Красный пункт в любой
        из двух частей держит задачу незавершённой.
        """
        if self.task is None:
            raise AgentError("Активной задачи нет: проверять нечего.")
        з = self.task
        пункты: list[dict[str, str]] = []

        def пункт(имя: str, ок: bool, пояснение: str) -> None:
            пункты.append({"пункт": имя, "итог": "прошло" if ок else "не прошло",
                           "пояснение": пояснение, "кем": "код"})

        пункт("план утверждён", з.утверждение_актуально,
              "подпись под текущим планом есть" if з.утверждение_актуально
              else "плана нет или он правился после утверждения")
        осталось = [ш for ш in з.шаги if ш.состояние != "готов"]
        пункт("все шаги пройдены", bool(з.шаги) and not осталось,
              f"шагов {len(з.шаги)}, не доведено {len(осталось)}")
        пункт("результаты собраны", bool(з.collected),
              f"записей в рабочей памяти: {len(з.collected)}")
        открытые = [о for о in з.ответы if not о.get("ответ")]
        пункт("вопросов без ответа нет", not открытые and з.ожидание != "ответ-пользователя",
              "задача ничего не ждёт от человека" if з.ожидание != "ответ-пользователя"
              else з.ожидание_текст)

        модель = ""
        if ревизор and self.client is not None and з.collected:
            итог = self._ревизия(model_key)
            if итог:
                исход, пояснение, модель = итог
                пункты.append({
                    "пункт": "сделанное отвечает плану (по смыслу)",
                    "итог": исход, "пояснение": пояснение,
                    "кем": модель or "ревизор",
                })

        # Красным считается только явное «не прошло». Третий исход — «не
        # проверено»: ревизор не ответил или ответил неразбираемо. Он виден в
        # отчёте, но задачу не запирает: человек не может починить чужой сбой
        # провайдера, а вечно незавершаемая задача — худший из исходов.
        красные = [п for п in пункты if п["итог"] == "не прошло"]
        не_проверено = [п for п in пункты if п["итог"] == "не проверено"]
        отчёт = з.записать_валидацию({
            "вердикт": "прошла" if not красные else "не прошла",
            "пункты": пункты,
            "красных": len(красные),
            "не_проверено": len(не_проверено),
            "ревизор": модель,
        })
        self.memory.working.save(з)
        self.memory._log("шаг-задачи", WORKING, з.task_id,
                         f"проверка: {отчёт['вердикт']}", applied=True,
                         reason=f"красных пунктов {len(красные)}")
        return отчёт

    def _ревизия(self, model_key: str = "") -> tuple[str, str, str] | None:
        """Смысловой вердикт модели: сходится ли сделанное с планом.

        Ревизору не показывают переписку — только план и результаты шагов. Его
        нечем уговаривать: уговорам поддаётся та модель, которую просят
        «посмотреть, всё ли в порядке», зная, чего от неё ждут.
        """
        з = self.task
        план = "\n".join(f"{i}. {ш}" for i, ш in enumerate(з.plan, 1)) or "плана нет"
        сделано = "\n\n".join(
            f"[{имя}]\n{(текст or '')[:1200]}" for имя, текст in з.collected.items()
        )[:6000]
        ключ = model_key or self.model_key or catalog.for_role("ревизор", offset=0)
        сообщения = [
            {"role": "system", "content": prompts.REVIEWER},
            {"role": "user",
             "content": f"ПЛАН ЗАДАЧИ:\n{план}\n\nЧТО СДЕЛАНО:\n{сделано}"},
        ]
        # Как и у судьи инвариантов, одна попытка подняться на ступень выше:
        # слабые модели регулярно отвечают не-JSON, и без эскалации смысловая
        # часть проверки пропадает целиком.
        последний = ""
        for попытка in range(2):
            try:
                ответ = self.client.call(ключ, сообщения, max_tokens=400,
                                         temperature=0.0, low_effort=True)
            except LLMError as exc:
                log.warning("ревизор не ответил (%s): %s", ключ, exc)
                последний = str(exc)[:160]
            else:
                данные = _разобрать_json((ответ.text or "").strip())
                # Разобранный JSON — ещё не вердикт: модель может вернуть
                # правильный по форме, но не тот ответ (чужую схему, пустой
                # объект). Вердиктом считается только тот, где есть ключ, о
                # котором спрашивали; иначе это «не проверено», а не «плохо».
                if данные is not None and "сходится" in данные:
                    сходится = bool(данные.get("сходится"))
                    пояснение = (данные.get("почему") or "").strip()[:300]
                    return (
                        "прошло" if сходится else "не прошло",
                        пояснение or ("сходится с планом" if сходится else "есть расхождения"),
                        ответ.model_key,
                    )
                последний = f"вердикт не разобран: {(ответ.text or '')[:120]}"
            if попытка == 0:
                следующая = catalog.escalate(ключ)
                if следующая != ключ:
                    ключ = следующая
                    continue
            break
        return "не проверено", f"ревизор не дал вердикта ({последний})", ключ

    # --- условия перехода ----------------------------------------------------

    def conditions(self) -> list[Условие]:
        return self.memory.all_conditions()

    def add_condition(self, условие: Условие) -> Условие:
        """Добавляет личное условие перехода. Базовые этим не тронуть."""
        try:
            return self.memory.long.conditions.add(условие)
        except TransitionConfigError as exc:
            raise AgentError(str(exc)) from exc

    def remove_condition(self, код: str) -> bool:
        return self.memory.long.conditions.remove(код)

    def finish_task(self, note: str = "") -> dict[str, Any]:
        """Завершает задачу: свёртка в журнал решений и очистка рабочей памяти."""
        if self.task is None:
            raise AgentError("Активной задачи нет.")
        if self.task.stage != DONE:
            # Завершение — такой же переход, как остальные, и проходит через те
            # же ворота. Раньше здесь была своя проверка «можно ли в done», и
            # она смотрела только на стрелку: задача закрывалась без всякой
            # проверки, лишь бы стадия была validation.
            self.transition(DONE, note or "задача завершена", кто=КОД)
        запись = self.memory.finish_task(self.task, note, model_key=self.model_key)
        self.task = None
        return запись

    # --- запись в память -----------------------------------------------------

    def remember(self, target: str, value: str, key: str = "", section: str = "",
                 reason: str = "") -> dict[str, Any]:
        """Явная запись в указанный слой. Модель не участвует."""
        try:
            return self.memory.remember_explicit(
                target, value, key=key, section=section, reason=reason,
                task_id=self.task.task_id if self.task else "",
            )
        except Exception as exc:  # LongTermError, WorkingMemoryError
            raise AgentError(str(exc)) from exc

    def remember_reply(self, role: str, text: str) -> None:
        """Кладёт реплику в краткосрочную память, если слой включён.

        Нужен исполнителю сценария: сам он ходит в ask() служебными вызовами,
        которые в диалог не пишут, а вопрос человека и итоговый ответ в истории
        разговора быть должны.
        """
        if SHORT in self.layers:
            self.memory.remember_message(
                role, text,
                task_id=self.task.task_id if self.task else "",
                stage=self.stage,
            )

    def remember_step(self, key: str, value: str) -> TaskState:
        if self.task is None:
            raise AgentError("Активной задачи нет: промежуточный результат некуда класть.")
        self.task = self.memory.remember_step(self.task, key, value)
        return self.task

    # --- основной цикл -------------------------------------------------------

    def checker(self) -> PreferenceChecker:
        """Проверка ответа по текущему профилю пользователя."""
        return PreferenceChecker(self.memory.long.profile.load())

    def ask(
        self,
        question: str,
        layers: set[str] | None = None,
        model_key: str = "",
        step_role: str = "",
        personal: bool = True,
        internal: bool = False,
    ) -> Answer:
        """Полный цикл: маршрутизация, сборка, вызов, проверка, запись.

        model_key и step_role задаёт исполнитель сценария: у каждого шага своя
        модель и своя роль, и общие настройки агента их не переопределяют.

        personal=False — промежуточный шаг сценария: настройки пользователя ни в
        промпт не идут, ни по ответу не проверяются.

        internal=True — вызов служебный, а не разговор с человеком: шаг сценария
        получает на вход машинный текст, собранный из результатов предыдущих
        шагов. Такой текст нельзя ни разбирать маршрутизатором, ни класть в
        диалог. Проверено, что бывает иначе: в базу знаний попадали записи вроде
        «Объяснительное сообщение о разделении работы между расширением и
        схемой» — маршрутизатор принял кусок ответа архитектора за факт,
        сказанный пользователем, и записал его навсегда.
        """
        question = (question or "").strip()
        if not question:
            raise AgentError("Пустой вопрос.")
        слои = set(layers) if layers is not None else set(self.layers)

        # Правило 5: может быть, в реплике есть что-то для долговременной памяти.
        # Если долговременный слой выключен, спрашивать маршрутизатор незачем.
        маршрут: Routing | None = None
        запись_маршрута: dict[str, Any] = {}
        if LONG in слои and not internal:
            маршрут, запись_маршрута = self.memory.route(question)

        # Правило 2: сама реплика — в краткосрочную память.
        if SHORT in слои and not internal:
            self.memory.remember_message(
                "user", question,
                task_id=self.task.task_id if self.task else "",
                stage=self.stage,
            )

        # Рубеж 1: не требует ли сам запрос нарушить инвариант. Отказ здесь
        # честнее и дешевле, чем «предложил, потом сам себя отругал»: токены не
        # тратятся, а пользователь не видит непоследовательности.
        if LONG in слои and not internal:
            конфликт = self.validator.check_request(question)
            if конфликт:
                return self._отказать(конфликт, "до вызова", question, слои)

        # Вопрос о самом инварианте — не попытка его нарушить. Объясняя, почему
        # проект не на Laravel, агент обязан назвать Laravel.
        объяснение = (self.validator.is_explanatory(question)
                      if LONG in слои and not internal else False)

        промпт = self.builder.build(question, self.task, слои, step_role=step_role,
                                    personal=personal)
        (ответ, попытки, нарушения, расхождения,
         эскалация, исправлено) = self._answer_within_rules(
            промпт, question, слои, model_key, step_role, personal, объяснение
        )

        применимые = (self.validator.applicable(question, ответ.text)
                      if LONG in слои else [])
        самоотчёт = self.validator.self_report(ответ.text)

        # Рубеж 3: смысловые инварианты. Суждение модели вызывается только когда
        # есть что судить и ответ дошёл до пользователя — на промежуточных шагах
        # сценария это лишний вызов на каждый шаг.
        вердикт = None
        if LONG in слои and personal and not нарушения and self.judge_semantic:
            вердикт = self.validator.judge(ответ.text, применимые)
            по_смыслу = self.validator.violations_from_judge(вердикт)
            if по_смыслу:
                нарушения = нарушения + по_смыслу

        отказ = None
        текст = ответ.text
        if нарушения:
            # Текст с нарушением наружу не выходит вовсе: «отказывается
            # предлагать» и «предложил с пометкой» — разные вещи. Сам ответ
            # остаётся в трейсе, разбирать его это не мешает.
            отказ = self.validator.refusal(нарушения, "после ответа", question)
            текст = отказ.текст()

        # Единственный канал, которым ассистент может повлиять на стадию:
        # строка «ПЕРЕХОД: done» в ответе. Просьба разбирается кодом и идёт
        # через те же ворота, что и команда человека, — поэтому «перепрыгнуть»
        # этап моделью нельзя, а попытка остаётся в журнале задачи.
        попытка: dict[str, Any] = {}
        if self.task is not None and not internal and not нарушения:
            попытка, текст = self._просьба_о_переходе(текст)

        мягкая: SoftResult | None = None
        if self.soft_check and not нарушения:
            мягкая = self.style.check(ответ.text)

        if SHORT in слои and not internal:
            self.memory.remember_message(
                "assistant", текст,
                task_id=self.task.task_id if self.task else "",
                stage=self.stage,
                tokens=ответ.total_tokens, cost=ответ.cost,
            )

        return Answer(
            text=текст,
            prompt=промпт,
            usage=ответ.to_dict(),
            attempts=попытки,
            violations=нарушения,
            deviations=расхождения,
            blocked=bool(нарушения),
            escalated_to=эскалация,
            refusal=отказ,
            применимые=[и.код for и in применимые],
            исправлено=исправлено,
            самоотчёт=самоотчёт,
            вердикт=вердикт,
            routing=маршрут,
            routing_entry=запись_маршрута,
            soft=мягкая,
            переход=попытка,
        )

    def _просьба_о_переходе(self, текст: str) -> tuple[dict[str, Any], str]:
        """Разбирает просьбу модели сменить стадию и отвечает на неё воротами.

        Наружу уходит не служебная строка, а результат: либо отметка о переходе,
        либо тот же отказ, который получил бы человек. Так «реакция ассистента»
        на недопустимый переход не зависит от настроения модели — её пишет код.
        """
        стадия = просьба_о_переходе(текст)
        if not стадия:
            return {}, текст
        чистый = убрать_маркер(текст)
        вердикт = self.ворота.перевести(self.task, стадия, МОДЕЛЬ,
                                        "по просьбе ассистента")
        self.memory.working.save(self.task)
        if вердикт.можно:
            основания = "; ".join(f"{п.код}: {п.видно}" for п in вердикт.проверки)
            примечание = (
                f"Стадия задачи переведена по просьбе ассистента: "
                f"{вердикт.откуда} → {вердикт.куда}."
                + (f"\nУсловия перехода проверены — {основания}." if основания else "")
            )
            self.memory._log("шаг-задачи", WORKING, self.task.task_id,
                             f"-> {стадия}", applied=True,
                             reason="переход по просьбе ассистента")
        else:
            примечание = (
                f"Ассистент попросил перейти на стадию «{стадия}». Запрос отклонён.\n\n"
                + вердикт.отказ.текст()
            )
            self.memory._log("шаг-задачи", WORKING, self.task.task_id,
                             f"-X-> {стадия}", applied=False,
                             reason=f"переход отклонён (модель): {вердикт.причина}")
        return вердикт.to_dict(), (чистый + "\n\n" + примечание).strip()

    def _отказать(self, нарушения: list[Violation], когда: str, question: str,
                  слои: set[str]) -> Answer:
        """Собирает ответ-отказ, не обращаясь к модели.

        Отказ пишет код, а не модель: тогда он одинаков при каждом конфликте,
        называет конкретный инвариант, его обоснование и допустимую
        альтернативу — всё это заранее записано в самом инварианте.
        """
        отказ = self.validator.refusal(нарушения, когда, question)
        промпт = self.builder.build(question, self.task, слои)
        if SHORT in слои:
            self.memory.remember_message("assistant", отказ.текст(),
                                         task_id=self.task.task_id if self.task else "",
                                         stage=self.stage)
        self.memory._log("явное-указание", LONG, "инварианты",
                         "; ".join(н.код for н in нарушения), applied=False,
                         reason=f"отказ {когда}: запрос требует нарушить инвариант")
        return Answer(
            text=отказ.текст(), prompt=промпт, attempts=0,
            violations=нарушения, blocked=True, refusal=отказ,
            применимые=[н.код for н in нарушения],
        )

    def _answer_within_rules(
        self,
        prompt: BuiltPrompt,
        question: str,
        layers: set[str],
        model_key: str = "",
        step_role: str = "",
        personal: bool = True,
        explanatory: bool = False,
    ) -> tuple[Reply, int, list[Violation], list[Deviation], str, list[str]]:
        """Получает ответ, который укладывается и в инварианты, и в профиль.

        Проверок две, и они разной силы. Инвариант — запрет проекта: ответ,
        который его нарушает, отдавать нельзя, и если переделать не удалось,
        агент честно говорит об этом. Расхождение с профилем — это «не так, как
        просил пользователь»: повторить стоит, но ответ по существу верен, и
        отдать его лучше, чем не отдать ничего.

        Лестница повторов общая: сначала просим переделать ту же модель, и
        только если не помогло — поднимаемся на ступень. Прыгать сразу на самую
        дорогую незачем, чаще всего хватает напоминания.
        """
        ключ_модели = model_key or self.model_for(prompt.stage)
        сообщения = prompt.messages
        эскалация = ""
        нарушения: list[Violation] = []
        расхождения: list[Deviation] = []
        # Нарушения, которые были на ранних попытках и ушли после напоминания.
        # Без этого отчёт не отличает «конфликта не было» от «конфликт был, и
        # агент переписал ответ сам» — а это разные исходы.
        исправлено: list[str] = []

        for попытка in range(1, MAX_ATTEMPTS + 1):
            try:
                ответ = self.client.call(ключ_модели, сообщения)
            except LLMError as exc:
                raise AgentError(str(exc)) from exc

            нарушения = (self.validator.check(ответ.text, explanatory)
                         if LONG in layers else [])
            # Самоотчёт требуется только от ответа, который увидит человек:
            # заставлять промежуточный шаг сценария перечислять инварианты —
            # значит тратить его выходные токены на служебную строку.
            # Самоотчёт стоит последней строкой ответа, поэтому обрезанный по
            # лимиту токенов ответ не может его содержать в принципе. Требовать
            # его тут — значит трижды переспросить и трижды получить тот же
            # обрубок. Именно так и вышло на первом живом прогоне.
            обрезан = ответ.finish_reason == "length"
            пропущено = (
                self.validator.check_self_report(
                    ответ.text, self.validator.applicable(question, ответ.text))
                if LONG in layers and personal and self.require_self_report
                and not обрезан else []
            )
            if обрезан and self.require_self_report:
                log.warning("Ответ обрезан по длине — самоотчёт не требуем")
            расхождения = (
                self.checker().check(ответ.text)
                if personal and self.follow_profile and LONG in layers else []
            )
            жёсткие = [о for о in расхождения if о.hard]
            if not нарушения and not жёсткие and not пропущено:
                return ответ, попытка, [], расхождения, эскалация, исправлено

            log.warning(
                "Попытка %d: нарушений %d, расхождений с профилем %d, "
                "пропущено в самоотчёте %d",
                попытка, len(нарушения), len(жёсткие), len(пропущено),
            )
            исправлено.extend(н.код for н in нарушения if н.код not in исправлено)
            if попытка == MAX_ATTEMPTS:
                break

            напоминания = []
            if нарушения:
                напоминания.append(self.validator.reminder(нарушения))
            if жёсткие:
                напоминания.append(self.checker().reminder(жёсткие))
            if пропущено:
                напоминания.append(self.validator.self_report_reminder(пропущено))
            повтор = self.builder.build(
                question, self.task, layers,
                extra_note="\n\n".join(напоминания), step_role=step_role,
                personal=personal,
            )
            сообщения = повтор.messages
            # Эскалация — только из-за инварианта. Расхождение с профилем
            # косметическое: платить за ответ вчетверо дороже потому, что он на
            # двадцать слов длиннее просимого, — плохая сделка. Повторяем на той
            # же модели: напоминание обычно помогает.
            if попытка >= 2 and нарушения:
                следующая = catalog.escalate(ключ_модели)
                if следующая != ключ_модели:
                    ключ_модели, эскалация = следующая, следующая

        # То, что осталось нарушенным, исправленным не считается.
        осталось = {н.код for н in нарушения}
        исправлено = [к for к in исправлено if к not in осталось]
        return ответ, MAX_ATTEMPTS, нарушения, расхождения, эскалация, исправлено

    # --- планирование --------------------------------------------------------

    def plan(self, note: str = "") -> Answer:
        """Просит модель составить план и кладёт его в рабочую память.

        Это единственное место, где ответ модели превращается в структуру, а не
        остаётся текстом: план — рабочие данные задачи, и жить он должен в
        рабочей памяти, а не в переписке.
        """
        if self.task is None:
            raise AgentError("Активной задачи нет: планировать нечего.")
        if self.task.stage != PLANNING:
            raise AgentError(
                f"План составляется на стадии planning, а задача сейчас в «{self.task.stage}»."
            )
        вопрос = (
            f"Составь план задачи «{self.task.title or self.task.task_id}». "
            + (note or "")
            + " Дай нумерованный список шагов, по одному шагу в строке, без пояснений между ними."
        )
        ответ = self.ask(вопрос)
        шаги = [ш.strip() for ш in _ШАГ_ПЛАНА.findall(ответ.text)][:12]
        if шаги:
            self.task = self.memory.remember_plan(self.task, шаги)
        return ответ

    # --- профиль пользователя ------------------------------------------------

    def profile(self) -> dict[str, Any]:
        return self.memory.long.profile.load()

    def needs_setup(self) -> bool:
        """Профиль ещё не настраивали — стоит предложить мастер."""
        return interview.needs_setup(self.profile())

    @staticmethod
    def setup_questions() -> list[dict[str, Any]]:
        return interview.questions()

    def setup(self, answers: dict[str, Any]) -> dict[str, Any]:
        """Применяет ответы мастера настройки к профилю."""
        try:
            профиль = interview.apply(self.profile(), answers)
        except preferences.PreferenceError as exc:
            raise AgentError(str(exc)) from exc
        self.memory.long.profile.save(профиль)
        self.memory._log("явное-указание", LONG, "профиль",
                         preferences.summary(профиль), applied=True,
                         reason="мастер настройки пройден")
        return профиль

    def use_template(self, name: str) -> dict[str, Any]:
        """Берёт готовую заготовку профиля целиком."""
        try:
            профиль = interview.from_template(name, self.profile())
        except preferences.PreferenceError as exc:
            raise AgentError(str(exc)) from exc
        self.memory.long.profile.save(профиль)
        self.memory._log("явное-указание", LONG, "профиль",
                         f"заготовка «{name}»: {preferences.summary(профиль)}",
                         applied=True, reason="выбрана готовая заготовка профиля")
        return профиль

    def set_preference(self, section: str, key: str, value: Any) -> dict[str, Any]:
        """Правит одно предпочтение; недопустимое значение отклоняется."""
        try:
            return self.memory.long.profile.update(section, key, value)
        except Exception as exc:
            raise AgentError(str(exc)) from exc

    # --- инварианты ------------------------------------------------------------

    def invariants(self) -> list:
        """Проектные инварианты плюс личные из профиля."""
        return self.memory.all_invariants()

    def add_invariant(self, инвариант, личный: bool = False):
        """Заводит инвариант. Личный ложится в профиль, проектный — в общий файл."""
        from agent.invariants import ЛИЧНЫЙ, InvariantError
        try:
            if личный:
                инвариант.уровень = ЛИЧНЫЙ
                инвариант.validate()
                профиль = self.memory.long.profile.load()
                прочие = [и for и in (профиль.get("инварианты") or [])
                          if и.get("код", "").lower() != инвариант.код.lower()]
                профиль["инварианты"] = прочие + [инвариант.to_dict()]
                self.memory.long.profile.save(профиль)
            else:
                self.memory.invariants.add(инвариант)
        except InvariantError as exc:
            raise AgentError(str(exc)) from exc
        self.memory._log("явное-указание", LONG, "инварианты", инвариант.правило,
                         applied=True,
                         reason=f"заведён {инвариант.уровень} инвариант «{инвариант.код}»")
        return инвариант

    def remove_invariant(self, код: str) -> bool:
        """Снимает инвариант. Личный — из профиля, проектный — из общего файла."""
        снят = self.memory.invariants.remove(код)
        профиль = self.memory.long.profile.load()
        личные = профиль.get("инварианты") or []
        осталось = [и for и in личные if и.get("код", "").lower() != (код or "").lower()]
        if len(осталось) != len(личные):
            профиль["инварианты"] = осталось
            self.memory.long.profile.save(профиль)
            снят = True
        if снят:
            self.memory._log("явное-указание", LONG, "инварианты", код, applied=True,
                             reason="инвариант снят")
        return снят

    def promote_decision(self, номер: int, код: str = "", вместо: str = ""):
        """Возводит запись журнала решений в инвариант.

        Замыкает то, ради чего журнал и ведётся: решение перестаёт быть заметкой
        на память и становится ограничением, которое нельзя обойти. Обоснованием
        отказа служит та самая причина, по которой решение приняли.
        """
        from agent.invariants import from_decision, InvariantError
        записи = self.memory.long.decisions.all()
        найдена = next((з for з in записи if int(з.get("id", 0)) == int(номер)), None)
        if найдена is None:
            есть = ", ".join(str(з.get("id")) for з in записи) or "журнал пуст"
            raise AgentError(f"В журнале решений нет записи №{номер}. Есть: {есть}.")
        try:
            инвариант = from_decision(найдена, код=код, вместо=вместо)
        except InvariantError as exc:
            raise AgentError(str(exc)) from exc
        return self.add_invariant(инвариант)

    # --- сценарии --------------------------------------------------------------

    def scenarios(self) -> list[Scenario]:
        return self.memory.long.scenarios.all()

    def match_scenario(self, query: str) -> Scenario | None:
        """Есть ли сценарий, чей триггер сработал на этом запросе."""
        return self.memory.long.scenarios.match(query)

    def add_scenario(self, scenario: Scenario) -> Scenario:
        try:
            итог = self.memory.long.scenarios.add(scenario)
        except Exception as exc:
            raise AgentError(str(exc)) from exc
        self.memory._log("явное-указание", LONG, "сценарии", итог.digest(), applied=True,
                         reason="сценарий добавлен пользователем")
        return итог

    def remove_scenario(self, name: str) -> bool:
        return self.memory.long.scenarios.remove(name)

    def run_scenario(
        self,
        query: str,
        name: str = "",
        on_step: Any = None,
        finish: bool = True,
        on_result: Any = None,
        по_шагам: bool = False,
    ) -> RunResult:
        """Исполняет сценарий: каждый шаг — свой агент, своя модель, своя стадия."""
        сценарий = (
            self.memory.long.scenarios.get(name) if name else self.match_scenario(query)
        )
        if сценарий is None:
            подсказка = (
                f"Нет сценария «{name}»." if name
                else "Ни один сценарий не сработал на этом запросе."
            )
            есть = ", ".join(с.имя for с in self.scenarios()) or "ни одного"
            raise AgentError(f"{подсказка} Заведено сценариев: {есть}.")
        try:
            return self.runner.run(сценарий, query, on_step=on_step, finish=finish,
                                   on_result=on_result, по_шагам=по_шагам)
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError(f"Сценарий «{сценарий.имя}» прервался: {exc}") from exc

    # --- сводка --------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """Состояние агента для интерфейсов."""
        модель = self.model_for()
        описание = catalog.get(модель)
        return {
            "model_key": модель,
            "model_label": описание.label,
            "model_fixed": bool(self.model_key),
            "provider": описание.provider,
            "free": описание.free,
            "user_id": self.user_id,
            "session": self.session,
            "layers": sorted(self.layers),
            "router_mode": self.memory.router_mode,
            "threshold": self.memory.threshold,
            "task": self.task.to_dict() if self.task else None,
            "stage": self.stage,
            "allowed": list(self.task.allowed()) if self.task else [],
            "soft_check": self.soft_check,
            "follow_profile": self.follow_profile,
            "judge_semantic": self.judge_semantic,
            "require_self_report": self.require_self_report,
            "инвариантов": len(self.memory.all_invariants()),
            "profile": preferences.summary(self.profile()),
            "needs_setup": self.needs_setup(),
            "scenarios": [с.digest() for с in self.scenarios()],
            "spent": dict(self.client.spent),
        }

    def stats(self) -> dict[str, Any]:
        return self.memory.stats()

    def journal(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.memory.journal(limit)

    def files(self) -> dict[str, str]:
        return self.memory.files()

    def tasks(self) -> list[dict[str, Any]]:
        return self.memory.working.tasks()

    def close(self) -> None:
        self.client.close()
