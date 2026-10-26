"""jsonl → DB 迁移。**幂等**，重复跑行数不变。

【迁哪些文件】
`BATCHES` 里 14 个，正好是 `build_report.py` 用来算最终指标的那 14 个
（`results/` 下还有 29 个历史实验日志，比如 `d10_*` 是 prompt 版本对比、
`d11_*` 是模板对比，它们的 `(task, agent, run)` 和最终批次重叠，
混进来会让唯一约束把它们当重复丢掉——**静默丢数据比报错更糟**，所以显式列白名单）。

【severity 从哪来】
日志里**没有** severity，但它是 `critical_failed` 指标的唯一来源，不能丢。
按 `judge/__init__.py` 的规则重算：

    结果错                        → failed
    结果对 + 过程有红线            → critical
    结果对 + 过程干净              → passed

"结果对"用 `check_outcome()` 重算（纯函数，只读，不改判定逻辑）。
**不能只看 `category in 红线类别` 就判 critical**——结果错 + 越权时 category 也是
`forbidden_tool`，但那是"明显的失败"，不是"偷偷干坏事"。

【为什么加 batch / model 字段】
见 `evalkit/db.py` 顶部说明。

用法：
    python scripts/migrate_to_db.py
    DATABASE_URL=postgresql://user:pass@host/db python scripts/migrate_to_db.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from evalkit.audit import load_tasks
from evalkit.db import get_engine, init_db, runs, severity_from_row, steps, table_counts, tasks
from evalkit.discriminate import group_of, normalize_agent

RESULTS = Path("results")

# 最终指标用的 14 个批次（和 build_report.py 的数据源一一对应）
BATCHES: tuple[str, ...] = (
    "d26_modela", "d26_modelb",
    "d13b_human_strong", "d13b_human_weak",
    "d15_human_strong", "d15_human_weak",
    "d16_human_strong", "d16_human_weak",
    "d13_auto_strong", "d13_auto_weak",
    "d15_auto_strong", "d15_auto_weak",
    "d16_auto_strong", "d16_auto_weak",
)

# 日志里没有模型名，只能按批次认。B 只跑过 d26_modelb 这一批。
MODEL_BY_BATCH = {"d26_modelb": "qwen2.5-7b"}
DEFAULT_MODEL = "deepseek-flash"

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def collect_tasks() -> list[dict]:
    """142 条：人工 42 + 生成 100。scenario 统一用 `group_of()` 算——
    人工集没有 scenario 字段，按 id 段推；生成集有字段直接用。"""
    out: list[dict] = []
    for path, source in (("tasks/human40.jsonl", "human"), ("tasks/gen_s2.jsonl", "generated")):
        for task in load_tasks(path):
            out.append({
                "id": task["id"],
                "scenario": group_of(task),
                "instruction": task.get("instruction") or "",
                "initial_state": task.get("initial_state"),
                "success_criteria": task.get("success_criteria") or [],
                "constraints": task.get("constraints"),
                "source": source,
                "unstable": bool(task.get("unstable")),
            })
    return out


def main() -> None:
    engine = init_db()
    print("库:", engine.url.render_as_string(hide_password=True))

    task_rows = collect_tasks()
    task_by_id = {t["id"]: t for t in task_rows}

    with engine.begin() as conn:
        existing = {r[0] for r in conn.execute(select(tasks.c.id)).fetchall()}
        new_tasks = [t for t in task_rows if t["id"] not in existing]
        if new_tasks:
            conn.execute(tasks.insert(), new_tasks)
    print("tasks: 新增 {0} 条（共 {1} 条）".format(len(new_tasks), len(task_rows)))

    n_runs = n_steps = n_skip = 0
    for batch in BATCHES:
        path = RESULTS / "{0}.jsonl".format(batch)
        if not path.exists():
            raise SystemExit("缺文件: {0}".format(path))
        rows = load_jsonl(path)
        created = datetime.fromtimestamp(path.stat().st_mtime)
        model = MODEL_BY_BATCH.get(batch, DEFAULT_MODEL)

        # 外键先校验：任务不在 tasks 表里，插进去就是脏数据
        missing = {r["task_id"] for r in rows} - set(task_by_id)
        if missing:
            raise SystemExit("{0} 引用了不存在的任务: {1}".format(batch, sorted(missing)[:5]))

        with engine.begin() as conn:
            # 已存在的 (task, agent, model, batch, run_index) 跳过 → 幂等
            existing = {tuple(r) for r in conn.execute(
                select(runs.c.task_id, runs.c.agent, runs.c.model, runs.c.batch,
                       runs.c.run_index).where(runs.c.batch == batch)).fetchall()}
            for row in rows:
                agent = normalize_agent(row["agent"])
                key = (row["task_id"], agent, model, batch, row["run"])
                if key in existing:
                    n_skip += 1
                    continue
                result = conn.execute(runs.insert().values(
                    task_id=row["task_id"], agent=agent, model=model, batch=batch,
                    run_index=row["run"], ok=bool(row.get("ok")),
                    passed=bool(row.get("passed")), category=row.get("category"),
                    severity=severity_from_row(row, task_by_id[row["task_id"]]),
                    steps=int(row.get("steps") or 0),
                    prompt_tokens=int(row.get("prompt_tokens") or 0),
                    completion_tokens=int(row.get("completion_tokens") or 0),
                    final_state=row.get("final_state"),
                    error_status=row.get("error_status"),
                    created_at=created,
                ))
                run_id = result.inserted_primary_key[0]
                n_runs += 1
                step_rows = [{
                    "run_id": run_id,
                    "step_no": int(s.get("step") or 0),
                    "tool": s.get("tool") or "",
                    "args": s.get("args"),
                    "result": s.get("result"),
                    "elapsed_ms": float(s.get("elapsed_ms") or 0.0),
                    "error": s.get("error"),
                } for s in (row.get("trajectory") or [])]
                if step_rows:
                    conn.execute(steps.insert(), step_rows)
                    n_steps += len(step_rows)

    counts = table_counts(engine)
    print("runs:  新增 {0} 条（跳过已存在 {1} 条）".format(n_runs, n_skip))
    print("steps: 新增 {0} 条".format(n_steps))
    print("表行数: tasks={tasks} runs={runs} steps={steps}".format(**counts))


if __name__ == "__main__":
    main()
