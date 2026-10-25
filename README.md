# LLM Agent 评测平台

给「会调工具的 Agent」做评测：任务集 → 跑任务 → 三层判定（结果 / 过程 / 效率）→ 指标 → 门禁 → 报告页 / 平台。

```
evalkit/         评测库（环境、循环、判定、指标、生成器、质检、报告）
server/          平台（FastAPI + 单页前端）
tasks/           任务集（人工 42 条 + 自动生成 100 条 + 探针 5 条）
results/         跑出来的日志、报告、perf 数据
docs/            契约、任务格式、场景说明、开发日志（踩坑记录）
tests/           217 条测试，全部离线（FakeClient / MockTransport）
```

## 怎么跑

```bash
# 测试（不需要 API key，不联网）
python -m pytest tests -q

# 跑一批任务并出报告
python -m evalkit.build_report          # → results/report.html（双击可开）

# 起平台
python -m uvicorn server.app:app --host 127.0.0.1 --port 8010
# 浏览器打开 http://127.0.0.1:8010/
```

## 已知局限

- **Docker 未验证**。仓库提供了 `Dockerfile`（`python:3.11-slim` + 单阶段 + `CMD pytest`），
  但本机 Docker 引擎依赖 WSL2，`wsl.exe` 被安全策略拦，**未执行 `docker build`**。
  依赖清单已在干净 venv 验证过（`pip install -r requirements.txt` + `pytest` → 217 passed）。
  要复现请在能跑 Docker 的环境执行：

  ```bash
  docker build -t evalkit .
  docker run --rm evalkit
  ```

- **平台跑的是单次任务**，不带 repeat。同一个任务连跑两次可能一次通过一次失败，
  页面上的结果就是那一次的结果；报告里的数字是多次跑的均值，两者不是一回事。

- **两种配置只差一个工具**（`full` 全给 / `no_order_info` 去掉 `get_order`）。
  这是为了控制变量，不代表"强/弱"——缺信息不等于弱，见 `docs/契约.md` 第七节。

- **门禁当前是 FAIL**：要做事组 gap 0.088，门槛 0.106（N=171）。
  要判定这个幅度真实存在需要 N≈248，现在样本不够。详见 `results/report.html`。

- **日志没存 `final_content`**，所以 `hallucinated_result` / `premature_finish`
  两类无法用已存轨迹重判（`scripts/rejudge_check.py` 会跳过这 5 条）。

- **成本只报 token，不报金额**。单价受峰谷定价和缓存命中率影响，同一批实验换个时段
  跑金额就不同，复现不出同一个数。
