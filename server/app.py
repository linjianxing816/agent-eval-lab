"""D19 平台后端：三个接口 + 一个页面。

**只做接线，不做逻辑**：跑任务调 `run_task`，判定调 `judge_task`，
Agent 配置直接用 `discriminate.AGENTS`。任何"顺手优化一下"都会让平台跑出来的结果
和报告里的数字对不上。

接口：
    GET  /api/runs          results/ 下的实验列表
    GET  /api/runs/{name}   单次实验详情（矩阵 / 门禁 / 统计）
    POST /api/run           跑一条任务，同步返回完整轨迹 + 判定
    GET  /api/tasks         任务列表（前端选择器要用；比"三个接口"多这一个）
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from evalkit.agent.loop import run_task
from evalkit.audit import load_tasks
from evalkit.client import DeepSeekClient, load_settings
from evalkit.discriminate import AGENTS
from evalkit.env.sandbox import Sandbox
from evalkit.env.state import EnvState
from evalkit.judge import judge_task

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STATIC = Path(__file__).resolve().parent / "static"
TASK_FILES = ("tasks/human40.jsonl", "tasks/gen_s2.jsonl")

app = FastAPI(title="LLM Agent 评测平台")

# 后端并发锁。前端那把锁只在浏览器里有效——直接打 API、或者开两个标签页，
# 就会同时发两次请求，一起打模型 API（很容易 429，而且两条轨迹会互相污染上下文）。
_run_lock = asyncio.Lock()


class RunRequest(BaseModel):
    task_id: str
    agent: str = "full"


def _load_all_tasks() -> dict[str, dict]:
    tasks: dict[str, dict] = {}
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
    """results/ 下的 .json 实验列表。

    不建数据库：文件就是真相。加了数据库之后，"报告里那个数"和"平台里那个数"
    会变成两份数据，迟早对不上。
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(RESULTS.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        rows.append({
            "name": path.stem,
            "bytes": path.stat().st_size,
            "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime)),
            "has_gate": isinstance(data, dict) and "gate" in data,
            "has_matrix": isinstance(data, dict) and bool(data.get("matrix")),
            "gate_status": (data.get("gate") or {}).get("status") if isinstance(data, dict) else None,
        })
    return {"runs": rows}


@app.get("/api/runs/{name}")
def get_run(name: str) -> dict[str, Any]:
    path = RESULTS / (name + ".json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="没有这个实验: " + name)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="这个文件不是对象结构，前端渲染不了")
    return {"name": name, "data": data}


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
    client = DeepSeekClient(load_settings(), thinking=False, max_tokens=1024, timeout=90.0)
    state = EnvState(task.get("initial_state"))
    sandbox = Sandbox(state)

    started = time.perf_counter()
    result = run_task(client, sandbox, instruction=task["instruction"],
                      system_prompt=agent["system_prompt"], tools=agent["tools"],
                      max_steps=(task.get("constraints") or {}).get("max_steps", 12))
    wall_ms = (time.perf_counter() - started) * 1000
    verdict = judge_task(task, state.data, result, sandbox.trajectory())

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
    }
