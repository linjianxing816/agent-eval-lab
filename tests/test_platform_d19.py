"""D19 平台接口测试。全部离线：跑任务那条把 DeepSeekClient 换成 FakeClient。"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from evalkit.client import FakeClient
from server import app as platform


@pytest.fixture()
def client():
    return TestClient(platform.app)


def _tool_call(name: str, args: dict, call_id: str = "c1") -> dict:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def test_runs_list_contains_report_data(client):
    body = client.get("/api/runs").json()
    names = {r["name"] for r in body["runs"]}
    assert "report_data" in names
    row = next(r for r in body["runs"] if r["name"] == "report_data")
    assert row["has_gate"] and row["has_matrix"]
    assert row["gate_status"] in ("PASS", "FAIL")


def test_run_detail_returns_matrix_and_gate(client):
    body = client.get("/api/runs/report_data").json()
    assert body["name"] == "report_data"
    assert "gate" in body["data"] and "matrix" in body["data"]
    # 轨迹不导出：1.3MB 里 99% 是轨迹，平台现场跑任务，不需要
    assert "trajectories" not in body["data"]


def test_unknown_run_is_404(client):
    assert client.get("/api/runs/不存在的实验").status_code == 404


def test_tasks_endpoint_lists_tasks_and_agents(client):
    body = client.get("/api/tasks").json()
    assert len(body["tasks"]) >= 100          # human40 42 + gen_s2 100，有重复 id 会少
    assert {a["name"] for a in body["agents"]} >= {"full", "no_order_info"}
    assert all("id" in t and "instruction" in t for t in body["tasks"])


def test_run_rejects_unknown_task(client):
    assert client.post("/api/run", json={"task_id": "没有这个", "agent": "full"}).status_code == 404


def test_run_rejects_unknown_agent(client):
    body = client.post("/api/run", json={"task_id": "task-031", "agent": "不存在的agent"})
    assert body.status_code == 400


def test_run_returns_trajectory_and_verdict(client, monkeypatch):
    # 只调 run_task + judge_task，所以把客户端换掉就能离线跑完整链路
    replies = [
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("get_order", {"order_id": "A123"})]},
        {"role": "assistant", "content": "A123 当前状态是 shipped，金额 299 元。"},
    ]
    monkeypatch.setattr(platform, "DeepSeekClient", lambda *a, **k: FakeClient(replies))

    body = client.post("/api/run", json={"task_id": "task-031", "agent": "full"}).json()
    assert body["task_id"] == "task-031"
    assert body["passed"] is True
    assert len(body["steps"]) == 1
    assert body["steps"][0]["tool"] == "get_order"
    assert body["steps"][0]["forbidden"] is False
    assert body["final_state"]["orders"]["A123"]["status"] == "shipped"
    assert body["prompt_tokens"] > 0


def test_run_marks_forbidden_step(client, monkeypatch):
    # task-001 的 forbid_tools 里有 delete_order，平台要能把那一步标红
    replies = [
        {"role": "assistant", "content": "",
         "tool_calls": [_tool_call("delete_order", {"order_id": "A123"})]},
        {"role": "assistant", "content": "已删除"},
    ]
    monkeypatch.setattr(platform, "DeepSeekClient", lambda *a, **k: FakeClient(replies))
    body = client.post("/api/run", json={"task_id": "task-001", "agent": "full"}).json()
    assert body["steps"][0]["forbidden"] is True
    assert body["passed"] is False
    assert body["category"] == "forbidden_tool"


def test_index_serves_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "LLM Agent 评测平台" in resp.text


def test_page_has_double_click_guard(client):
    # 连点两次会同时打 API —— 前端要有显式标志位，不能只靠 button.disabled
    html = client.get("/").text
    assert "let running = false" in html
    assert "if (running) return;" in html
    assert "$('taskSel').disabled = true" in html


def test_page_notes_single_run_randomness(client):
    # 平台跑单次任务，和报告里的多次均值不是一回事，页面上要写清
    html = client.get("/").text
    assert "单次运行，结果有随机性" in html


def test_readme_documents_docker_limitation():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert "已知局限" in text
    assert "docker build" in text and "wsl.exe" in text


def test_second_concurrent_run_is_rejected_with_429(client, monkeypatch):
    """前端锁只在浏览器里有效；直接打 API 连发两次必须被后端拦住。"""
    import threading
    import time as _time

    started = threading.Event()

    def slow_execute(payload):
        started.set()
        _time.sleep(0.4)
        return {"task_id": payload.task_id, "passed": True}

    monkeypatch.setattr(platform, "_execute_run", slow_execute)

    box = {}

    def first():
        box["first"] = client.post("/api/run", json={"task_id": "task-031", "agent": "full"})

    thread = threading.Thread(target=first)
    thread.start()
    assert started.wait(3), "第一个请求没启动"

    second = client.post("/api/run", json={"task_id": "task-031", "agent": "full"})
    thread.join(5)

    assert box["first"].status_code == 200
    assert second.status_code == 429
    assert "已有任务在运行" in second.json()["error"]


def test_model_b_settings_requires_all_three(tmp_path, monkeypatch):
    from evalkit.client import MissingConfigError, load_model_b_settings
    for key in ("MODEL_B_BASE_URL", "MODEL_B_API_KEY", "MODEL_B_MODEL"):
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    env.write_text("MODEL_B_BASE_URL=https://x\n", encoding="utf-8")
    with pytest.raises(MissingConfigError) as exc:
        load_model_b_settings(env_path=env)
    assert "MODEL_B_API_KEY" in str(exc.value)


def test_model_b_settings_reads_three_values(tmp_path, monkeypatch):
    from evalkit.client import load_model_b_settings
    for key in ("MODEL_B_BASE_URL", "MODEL_B_API_KEY", "MODEL_B_MODEL"):
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    env.write_text("MODEL_B_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1\n"
                   "MODEL_B_API_KEY=sk-test\nMODEL_B_MODEL=qwen-turbo\n", encoding="utf-8")
    settings = load_model_b_settings(env_path=env)
    assert settings.model == "qwen-turbo"
    assert settings.base_url.endswith("/v1")
