"""D19 平台后端：三个接口 + 一个页面。

**只做接线，不做逻辑**：跑任务调 `run_task`，判定调 `judge_task`，
Agent 配置直接用 `discriminate.AGENTS`。任何"顺手优化一下"都会让平台跑出来的结果
和报告里的数字对不上。

接口（**数据源已切到 DB**）：
    GET  /api/runs          DB 里的批次清单（跑了几次 / 多少任务 / 时间范围）+ 报告文件
    GET  /api/runs/{name}   批次统计（DB 现算）或报告详情（results/*.json）
    POST /api/run           跑一条任务，同步返回轨迹 + 判定，**结果写进 runs/steps 表**
    GET  /api/tasks         任务列表（DB 优先，缺了回退 tasks/*.jsonl）

**为什么列表改成读 DB**：以前扫 `results/*.json` 返回文件字节数——
字节数说明不了任何事，跑 90 次和跑 30 次的 jsonl 可能差不多大。
从 `runs` 表聚合才能回答"这批到底跑了多少次"。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from evalkit import db_queries
from evalkit.agent.loop import run_task
from evalkit.audit import load_tasks
from evalkit.client import DeepSeekClient, load_settings
from evalkit.db import get_engine, save_run
from evalkit.discriminate import AGENTS
from evalkit.env.sandbox import Sandbox
from evalkit.env.state import EnvState
from evalkit.judge import judge_task

logger = logging.getLogger("evalkit.server")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STATIC = Path(__file__).resolve().parent / "static"
TASK_FILES = ("tasks/human40.jsonl", "tasks/gen_s2.jsonl")
# 平台跑的任务单独一个批次，**不进定稿批次**（它不参与任何报告数字）
PLATFORM_BATCH = "platform"

app = FastAPI(title="LLM Agent 评测平台")

# 后端并发锁。前端那把锁只在浏览器里有效——直接打 API、或者开两个标签页，
# 就会同时发两次请求，一起打模型 API（很容易 429，而且两条轨迹会互相污染上下文）。
_run_lock = asyncio.Lock()


class RunRequest(BaseModel):
    task_id: str
    agent: str = "full"


def _load_all_tasks() -> dict[str, dict]:
    """任务定义：**DB 优先**，DB 没数据回退文件。

    返回完整 task dict（`POST /api/run` 要拿 `initial_state` / `success_criteria` 去执行），
    所以不能直接用 `db_queries` 里那种精简结构。
    """
    tasks: dict[str, dict] = {}
    try:
        rows = db_queries._run(get_engine(), """
            SELECT id, scenario, instruction, initial_state, success_criteria, constraints
            FROM tasks
        """)
        for tid, scenario, instruction, initial, criteria, constraints in rows:
            tasks[tid] = {
                "id": tid, "scenario": scenario, "instruction": instruction,
                "initial_state": db_queries._json_value(initial),
                "success_criteria": db_queries._json_value(criteria) or [],
                "constraints": db_queries._json_value(constraints),
            }
    except Exception as exc:                      # 库没建 / 驱动缺失 → 回退文件
        logger.warning("读 DB 的任务表失败，回退 tasks/*.jsonl: %s", exc)
    if tasks:
        return tasks
    for relative in TASK_FILES:
        path = ROOT / relative
        if path.exists():
            for task in load_tasks(relative):
                tasks[task["id"]] = task
    return tasks


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/tasks")
def list_tasks() -> dict[str, Any]:
    """任务列表。只给选择器需要的字段，不把 criteria 全塞过去。"""
    tasks = _load_all_tasks()
    return {
        "tasks": [
            {"id": t["id"], "instruction": t.get("instruction", ""),
             "scenario": t.get("scenario") or "", "max_steps": (t.get("constraints") or {}).get("max_steps", 12)}
            for t in tasks.values()
        ],
        "agents": [{"name": name, "tools": len(cfg["tools"])} for name, cfg in AGENTS.items()],
    }


@app.get("/api/runs")
def list_runs() -> dict[str, Any]:
    """实验列表：**DB 批次优先**，`results/` 下的报告文件排在后面。

    每条都带 `source` 区分来源（`db` = 执行批次，`file` = 报告 json）。
    批次没有门禁/矩阵，那两个字段给 null——前端显示"—"就行。
    """
    rows: list[dict[str, Any]] = []
    try:
        for batch in db_queries.list_batches():
            rows.append({
                "name": batch["name"],
                "runs": batch["runs"],
                "tasks": batch["tasks"],
                "passed": batch["passed"],
                "first": batch["first"],
                "last": batch["last"],
                "canonical": batch["canonical"],
                "source": "db",
                # 前端原有字段，批次没有对应的东西，给 null 让前端显示"—"
                "mtime": batch["first"],
                "gate_status": None,
                "has_gate": False,
                "has_matrix": False,
            })
    except Exception as exc:
        logger.warning("读 DB 批次失败，只列报告文件: %s", exc)

    for path in sorted(RESULTS.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        rows.append({
            "name": path.stem,
            "runs": None,
            "tasks": None,
            "passed": None,
            "first": None,
            "last": None,
            "canonical": False,
            "source": "file",
            "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime)),
            "gate_status": (data.get("gate") or {}).get("status") if isinstance(data, dict) else None,
            "has_gate": isinstance(data, dict) and "gate" in data,
            "has_matrix": isinstance(data, dict) and bool(data.get("matrix")),
        })
    return {"runs": rows}


@app.get("/api/runs/{name}")
def get_run(name: str) -> dict[str, Any]:
    """先当批次名查 DB，查不到再当文件名读 json。

    **顺序不能反**：批次名和文件名没有交集（`d26_modela` 没有同名 json），
    但先查 DB 才符合"平台读 DB"的定位；json 只是历史报告文件的兼容入口。
    """
    try:
        summary = db_queries.batch_summary(name)
    except Exception as exc:
        logger.warning("读 DB 批次失败，回退文件: %s", exc)
        summary = None
    if summary is not None:
        return {"name": name, "source": "db", "data": summary}

    path = RESULTS / (name + ".json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="没有这个批次或实验: " + name)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="这个文件不是对象结构，前端渲染不了")
    return {"name": name, "source": "file", "data": data}


@app.post("/api/run")
async def run_one(payload: RunRequest) -> Any:
    """真跑一条任务。同步返回——一条约 5–8 秒，前端显示 loading 就够，不做异步轮询。

    **用 async def + 线程池**：`run_task` 是阻塞的，直接在事件循环里跑会把整个服务卡住
    （连 429 都返回不了）。放到线程池里，锁才拦得住第二个请求。
    """
    if _run_lock.locked():
        return JSONResponse({"error": "已有任务在运行，请等待"}, status_code=429)
    async with _run_lock:
        return await run_in_threadpool(_execute_run, payload)


def _execute_run(payload: RunRequest) -> dict[str, Any]:
    tasks = _load_all_tasks()
    task = tasks.get(payload.task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="没有这个任务: " + payload.task_id)
    if payload.agent not in AGENTS:
        raise HTTPException(status_code=400, detail="没有这个 Agent: " + payload.agent)

    agent = AGENTS[payload.agent]
    settings = load_settings()
    client = DeepSeekClient(settings, thinking=False, max_tokens=1024, timeout=90.0)
    state = EnvState(task.get("initial_state"))
    sandbox = Sandbox(state)

    started = time.perf_counter()
    result = run_task(client, sandbox, instruction=task["instruction"],
                      system_prompt=agent["system_prompt"], tools=agent["tools"],
                      max_steps=(task.get("constraints") or {}).get("max_steps", 12))
    wall_ms = (time.perf_counter() - started) * 1000
    verdict = judge_task(task, state.data, result, sandbox.trajectory())

    # **写库失败不影响本次返回**：用户跑一次任务不该因为 DB 挂了就拿不到结果。
    # `stored=False` 会带回去，前端能看到"这次没存进库"。
    run_id, stored = None, False
    try:
        run_id = save_run(
            get_engine(),
            {
                "task_id": task["id"], "agent": payload.agent, "model": settings.model,
                "batch": PLATFORM_BATCH, "ok": True,
                "passed": bool(verdict["passed"]), "category": verdict["category"],
                "severity": verdict["severity"], "steps": result.steps,
                "prompt_tokens": sum(result.per_step_tokens),
                "completion_tokens": sum(result.per_step_completion_tokens),
                "final_state": state.data,
            },
            [{"step_no": s.step, "tool": s.tool, "args": s.args, "result": s.result,
              "elapsed_ms": float(s.elapsed_ms), "error": s.error}
             for s in sandbox.trajectory()],
        )
        stored = True
    except Exception as exc:
        logger.error("写 DB 失败（结果照常返回）: %s", exc)

    forbid = set((task.get("constraints") or {}).get("forbid_tools") or [])
    return {
        "task_id": task["id"],
        "agent": payload.agent,
        "instruction": task["instruction"],
        "criteria": task.get("success_criteria") or [],
        "passed": verdict["passed"],
        "severity": verdict["severity"],
        "category": verdict["category"],
        "detail": verdict["detail"],
        "warnings": verdict["warnings"],
        "steps": [
            {"step": step.step, "tool": step.tool, "args": step.args, "result": step.result,
             "error": step.error, "elapsed_ms": round(step.elapsed_ms, 2),
             "forbidden": step.tool in forbid}
            for step in sandbox.trajectory()
        ],
        "final_state": state.data,
        "final_content": (result.final_content or "")[:500],
        "prompt_tokens": sum(result.per_step_tokens),
        "completion_tokens": sum(result.per_step_completion_tokens),
        "wall_ms": round(wall_ms, 1),
        "exit_reason": result.exit_reason,
        "run_id": run_id,
        "stored": stored,
    }
