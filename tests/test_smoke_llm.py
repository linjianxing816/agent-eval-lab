"""D0 smoke 脚本的离线单测。

全程不联网：URL 拼接是纯函数；配置用临时 .env；
chat() 用 httpx.MockTransport 模拟 401，验证「报错不泄露 key」这条验收。
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import smoke_llm  # noqa: E402


# ---------- build_url：末尾带 / 与不带 / 两个边界 ----------

def test_build_url_without_trailing_slash():
    assert smoke_llm.build_url("https://api.deepseek.com") == (
        "https://api.deepseek.com/chat/completions"
    )


def test_build_url_with_trailing_slash():
    assert smoke_llm.build_url("https://api.deepseek.com/") == (
        "https://api.deepseek.com/chat/completions"
    )


def test_build_url_vendor_with_v1_level():
    # 通义这种要带 /v1 的层级，.env 负责，脚本只管拼接
    assert smoke_llm.build_url("https://dashscope.aliyuncs.com/compatible-mode/v1/") == (
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    )


def test_build_url_rejects_empty():
    with pytest.raises(smoke_llm.MissingConfigError):
        smoke_llm.build_url("   ")


def test_build_url_rejects_non_http():
    with pytest.raises(smoke_llm.MissingConfigError):
        smoke_llm.build_url("api.deepseek.com")


# ---------- load_settings：缺项报错，且消息里不含值 ----------

REQUIRED = smoke_llm.REQUIRED_KEYS


@pytest.fixture()
def clean_env(monkeypatch):
    for name in REQUIRED:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_load_settings_missing_key_names_variable_only(clean_env, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AI_BASE_URL=https://api.deepseek.com\nAI_MODEL=deepseek-chat\n",
        encoding="utf-8",
    )
    with pytest.raises(smoke_llm.MissingConfigError) as excinfo:
        smoke_llm.load_settings(env_file)
    message = str(excinfo.value)
    assert "AI_API_KEY" in message          # 只报变量名
    assert "sk-" not in message             # 不含任何疑似 key 的值


def test_load_settings_rejects_placeholder_key(clean_env, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AI_BASE_URL=https://api.deepseek.com\n"
        "AI_API_KEY=sk-在这里填你的key\n"
        "AI_MODEL=deepseek-chat\n",
        encoding="utf-8",
    )
    with pytest.raises(smoke_llm.MissingConfigError) as excinfo:
        smoke_llm.load_settings(env_file)
    assert "占位符" in str(excinfo.value)


def test_load_settings_reads_all_three(clean_env, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AI_BASE_URL=https://api.deepseek.com\n"
        "AI_API_KEY=sk-test-abc123\n"
        "AI_MODEL=deepseek-chat\n",
        encoding="utf-8",
    )
    settings = smoke_llm.load_settings(env_file)
    assert settings.model == "deepseek-chat"
    assert settings.base_url == "https://api.deepseek.com"
    # 不断言 key 内容本身，只确认它被读到了
    assert settings.api_key.startswith("sk-test-")


# ---------- chat：401 时报错绝不包含 key（MockTransport，不联网） ----------

def _settings() -> smoke_llm.Settings:
    return smoke_llm.Settings(
        base_url="https://api.deepseek.com",
        api_key="sk-secret-do-not-leak-000",
        model="deepseek-chat",
    )


def test_chat_401_error_does_not_leak_key():
    def handler(request: httpx.Request) -> httpx.Response:
        # 模拟厂商把 key 回显在错误 body 里的最坏情况
        return httpx.Response(
            401,
            json={"error": {"message": "Authentication Fails, Your api key: sk-secret-do-not-leak-000"}},
        )

    with pytest.raises(smoke_llm.ChatError) as excinfo:
        smoke_llm.chat(_settings(), "hi", transport=httpx.MockTransport(handler))
    message = str(excinfo.value)
    assert "401" in message
    assert "sk-secret-do-not-leak-000" not in message


def test_chat_timeout_raises_chat_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    with pytest.raises(smoke_llm.ChatError) as excinfo:
        smoke_llm.chat(_settings(), "hi", transport=httpx.MockTransport(handler))
    assert "超时" in str(excinfo.value)


def test_chat_parses_answer_and_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("Authorization", "")
        assert auth == "Bearer sk-secret-do-not-leak-000"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "回归测试是改代码后重跑旧用例。"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 20},
            },
        )

    result = smoke_llm.chat(_settings(), "hi", transport=httpx.MockTransport(handler))
    assert result.text.startswith("回归测试")
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 20
    assert result.elapsed_ms >= 0
