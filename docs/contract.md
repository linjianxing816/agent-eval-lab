# contract.md · LLM 评测与推理性能实验平台

> 这份文件是"我定的规则"，不是"AI 写的代码说明"。
> 实现和这里不一致，以这里为准。面试时这份文件就是你的讲稿底稿。

---

## 一、一句话定位

**一个实验平台，回答两类问题：题目本身靠不靠谱（质量），以及模型跑得快不快、贵不贵（性能）。**

三个模块：
1. **数据集构建与质检** —— 让模型生成测试用例，再用四个维度验证它们值不值得用
2. **效果评测与对比** —— 拿合格题集去测多个系统，对比过滤前后结论是否一致
3. **推理性能基准** —— 采集 TTFT / TPOT / 吞吐 / 成本，看质量与性能的取舍

**为什么不是"API 拼接"**：单纯生成 200 道题 + 调三个模型打分，确实是拼接。
但加上"定义什么叫好题（区分度）"、"验证题集能不能替代人工集"、"采集推理性能并做质量-成本权衡"，就有了实验设计和指标设计的内容。

**顺序**：先做模块 1+2（质量），跑出第一份真实报告；再做模块 3（性能）。

---

## 二、核心问题链

1. 人工写 60 条用例要两天，覆盖有限，还带着自己的偏好 → **能不能让模型生成？**
2. 生成 200 条，怎么知道哪些有用、哪些是垃圾？→ **得先定义"好用例"的标准**
3. 自动生成的集合，能替代人工写的集合吗？→ **做对比实验，看结论是否一致**

---

## 三、四个质量维度（这是项目的核心方法论）

一条自动生成的用例，要过四关才算"能用"。

| 维度 | 定义 | 怎么测 | 门槛 |
|---|---|---|---|
| **可执行性** | 格式合法、字段完整、能跑起来 | 自动校验（复用 `validate`） | 必须 100% |
| **可判定性** | `expected` 明确，没有歧义 | 同一模型独立判 3 次，看结论是否一致 | ≥ 85% |
| **正确性** | `expected` 本身是对的 | 人工抽检 20 条 + 双模型交叉验证 | ≥ 80% |
| **区分度** | 能不能区分好坏系统 | 强 prompt vs 弱 prompt 的通过率差 | 均值 ≥ 0.25 |

### 3.1 区分度（这是最有价值的一个指标）

**核心想法**：一条用例如果所有模型都过、或者都不过，它就是废的。**好用例的价值在于"能发现差异"。**

定义：
```
discrimination(sample) = |p_strong(sample) - p_weak(sample)|

其中 p = 同一用例跑 3 轮的通过率均值（消除单次波动）
strong = prompts/v2.txt（加严版）
weak   = prompts/v1.txt（基础版）
```

| 情况 | discrimination | 含义 |
|---|---|---|
| 强过、弱不过 | 1.0 | **最理想**：这条用例能抓住差异 |
| 强过、弱也过 | 0 | 太简单，测不出东西 |
| 强不过、弱不过 | 0 | 太难，或者本身有毛病 |
| 两者都不稳定 | 0.3–0.7 | 有区分，但结论不稳，要看 repeat |

**聚合指标**：
- 平均区分度 = 所有样本 `discrimination` 的均值
- **有效用例比例** = `discrimination ≥ 0.3` 的样本占比
- 区分度分布直方图（看是集中在 0 还是分散）

> **两个阈值别混（D8 最容易搞错）**：
> - **0.25** = 整个集合的**平均区分度**门槛（低于它说明这批题整体没区分力）
> - **0.3** = **单条用例**算不算"有效"的门槛（用来算有效用例比例）
>
> 一条题可以区分度 0.2（不算有效），但集合平均值仍可能过 0.25——这两个数回答的是不同问题。

**为什么要 repeat 3 次**：单次结果受 temperature 和网络影响，一条用例一次过、一次不过，你不能断定它有区分度。3 轮取均值才稳。

### 3.2 可判定性（歧义检测）

`expected` 有歧义，评分器就会误判——而且你永远不知道是模型错了还是题错了。

测法：**同一模型对同一条用例独立判 3 次（判 pass/fail），3 次结论不一致 = 有歧义。**

注意：这里判的是"模型的输出对不对"，不是"模型答得好不好"。做法是先用固定 prompt 跑 3 轮拿到 3 个输出，再用评分器分别判，看 3 个判定是否一致。

不一致的原因可能是：模型输出不稳定（temperature），也可能是断言本身模糊（比如 `contains` 只要求关键词，模型说了"不是 X"也命中）。**两种情况都要记，分开统计。**

---

## 四、生成器

### 4.1 输入：任务规格（spec）

生成用例不能只说"给我出几道题"，得给结构化描述：

```yaml
task: 从中文文本中抽取姓名、年龄、城市，输出 JSON
output_format: json
fields:
  name: string
  age: int
  city: string
constraints:
  - 年龄必须是整数，不要带单位
  - 城市必须是地级市
  - 信息不足时对应字段返回 null
types: [json_check, field_check]
difficulty_mix: {easy: 0.3, medium: 0.5, hard: 0.2}
adversarial_ratio: 0.25
n: 200
```

**spec 是手写的，不是模型生成的**。spec 定义"要测什么"，模型只负责"出题"。

### 4.2 三种生成策略（要对比）

| 策略 | 做法 | 预期 |
|---|---|---|
| **S1 单轮** | 一次让它生成 200 条 | 快，但重复多、难度分布失控 |
| **S2 分批** | 每批 20 条，分批生成 + 去重 | 重复少，分布可控 |
| **S3 反馈迭代** | 先跑一轮，把失败样本当"难例提示"再生成第二批 | 区分度最高，但成本翻倍 |

**这是项目里第二个实验**：三种策略的用例质量（四个维度）对比。

### 4.3 生成后必须做的三件事

1. **去重**：`input` 完全相同 → 直接删；`input` 高度相似（字符级 Jaccard > 0.85）→ 标记待人工确认
2. **格式校验**：复用 `validate`，不合格的直接丢
3. **来源标记**：每条用例加 `source: generated|human`，后面所有对比都靠这个字段分组

---

## 五、数据集格式

### 5.1 字段

```json
{"id":"gen-0001","type":"field_check","input":"张三今年30岁，住在杭州","expected":{"name":"string","age":"int","city":"string"},"difficulty":"easy","category":"extraction","adversarial":false,"source":"generated","spec":"extraction-v1","note":"基础三元组抽取"}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `id` | 是 | **全局唯一且稳定**，`diff` 靠它对齐 |
| `type` | 是 | `exact_match` / `json_check` / `field_check` / `contains` |
| `input` | 是 | 发给模型的内容 |
| `expected` | 视类型 | 见 5.2 |
| `mode` | 否 | 仅 `contains`：`all` / `any` |
| `exclude` | 否 | 仅 `contains`：命中即失败，优先级高于 `mode` |
| `adversarial` | 否 | 对抗样本标记 |
| `difficulty` | 否 | `easy` / `medium` / `hard` |
| `category` | 否 | 业务标签 |
| `source` | 是 | **`human` / `generated`**，对比实验的分组依据 |
| `spec` | 否 | 生成时用的 spec 名，用于追溯 |
| `note` | 否 | 这条想验证什么 |

**`source` 字段是整个对比实验的基础**，生成器必须自动填，人工集必须手填 `human`。

### 5.2 四类判定规则

#### `exact_match`
- `expected` 是字符串，判定 `normalize(输出) == normalize(expected)`
- 归一化 7 步，**顺序不能变**：
  1. 去首尾空白
  2. 去 markdown 包裹（`**`、`` ` ``）
  3. 去前缀"答案是 / 答：/ 答案："
  4. 全角转半角
  5. 去中英文标点
  6. 英文转小写
  7. 中文数字转阿拉伯（0–99）

#### `json_check`
- `expected` 为 `null`，判定能否解析
- **必须记两个值**：`raw_parse_ok`（原文直接解析）、`clean_parse_ok`（剥 ``` 围栏、取首 `{` 到末 `}` 后解析）
- 两者差值 = "会写 JSON 但爱加废话"的比例

#### `field_check`
- `expected` 是 `{字段名: 类型}`，类型：`string` / `int` / `float` / `bool` / `array` / `object` / `null`
- 全部字段存在且类型匹配才算通过
- **类型严格**：`"18"` 不算 `int`
- 分开记"字段缺失"和"类型错"，另算 `partial_score`

#### `contains`
- `expected` 是关键词列表，`mode=all/any`
- `exclude` 命中任一 → 直接失败，**优先级最高**

### 5.3 规模

| 集合 | 条数 | 用途 |
|---|---|---|
| 人工基线 | 60（exact 12 / json 16 / field 16 / contains 16，含 16 条对抗） | 对照组 |
| 自动生成 | 200 → 过滤后约 120–150 | 实验组 |
| 人工抽检 | 20（从自动集随机抽） | 验证正确性 |

---

## 六、失败分类

每条失败样本必须归一类，不许只记 `pass/fail`。

| 类别 | 含义 |
|---|---|
| `request_error` | 请求失败（超时 / 5xx / 网络） |
| `rate_limited` | 429 限流，**单独一类** |
| `empty_output` | 200 但内容为空 |
| `truncated` | 输出被 max_tokens 截断 |
| `invalid_json` | 无法解析为 JSON |
| `missing_field` | 字段缺失 |
| `wrong_type` | 字段类型错 |
| `wrong_value` | 类型对但值不对 |
| `assertion_failed` | 其他断言不通过 |

同时记 `assertion_detail`，一句人话，比如 `"期望 int，实际 string（'18岁'）"`。

---

## 七、CLI

```
# 生成
eval spec validate --spec specs/extraction.yaml
eval gen --spec specs/extraction.yaml --strategy s2 --n 200 --out data/gen_s2.jsonl

# 质量验证（四关）
eval audit --dataset data/gen_s2.jsonl                       # 可执行性 / 重复 / 分布
eval ambiguity --dataset data/gen_s2.jsonl --repeat 3         # 可判定性
eval discriminate --dataset data/gen_s2.jsonl \
     --strong prompts/v2.txt --weak prompts/v1.txt --repeat 3 # 区分度

# 评测
eval run --dataset data/gen_s2.jsonl --prompt-version v1 --concurrency 4 --repeat 3
eval run --dataset data/human60.jsonl --prompt-version v1 --concurrency 4 --repeat 3

# 对比与报告
eval diff results/<a> results/<b>
eval compare results/<a> results/<b> ...
eval report --run results/<id>
eval html --run results/<id>

# 门禁
eval gate --baseline baseline.json --run results/<id>
```

**`eval discriminate` 是最核心的命令**——它决定"这条用例值不值得留"。

### 7.1 平台层（薄服务，D17，可砍）

**定位**：生成流水线的控制台，不是"又一个评测平台"。**时间紧就整个砍掉，不影响主线成立。**

接口（如果做）：
```
POST /api/tasks                  提交 {spec, strategy, n} → {task_id}
GET  /api/tasks                  任务列表（状态 + 进度 done/total）
GET  /api/tasks/{id}             详情 + 四维质检结果
GET  /api/datasets               已有用例集
```

状态机：`pending → running → done | failed`，**没有取消、没有重试队列**。
并发：全局 `asyncio.Semaphore(2)`，目的是**防止连点把上游 API 打爆**，不是性能优化。

**硬约束**：`server.py` 只调 `runner.py` / `generator/`，**不复制任何生成或评测逻辑**。

**前端**（原生 HTML + fetch，无构建链）：提交区 + 任务列表 + 四个大数字 + 区分度分布图。

**明确不做**：登录 / 多用户 / 权限 / WebSocket / Celery / Redis / 在线编辑评分规则。

---

## 八、结果目录

```
results/<run_id>/
  config.json      spec、策略、模型、prompt_version、并发、repeat、数据集 hash、
                   单价、超时与重试、代码 commit
  samples.jsonl    每条：id、source、input、raw_output、判定、failure_category、
                   assertion_detail、耗时、token
  metrics.json     汇总（含按 source 分组、失败分类分布）
  report.md
  report.html
  diff.json / diff.md   （可选）
```

`run_id = YYYYMMDD-HHMM-<数据集hash前4位>`。**只增不改。**

**可追溯**：拿到任意 `run_id`，要能回答——哪版 spec、哪个策略、哪版 prompt、哪版数据集、哪版代码、花了多少钱。

---

## 九、指标口径

### 9.1 用例质量指标（生成集专用）

| 指标 | 公式 | 注意 |
|---|---|---|
| 可执行率 | 通过 validate 的条数 / 生成总数 | 分母是**生成总数**，不是过滤后的 |
| 去重率 | 被删重复条数 / 生成总数 | 反映生成策略的重复程度 |
| 可判定率 | 3 次判定一致的条数 / 参与检测条数 | 不一致要分"模型不稳"和"断言模糊" |
| 正确率 | 人工判定正确的条数 / 抽检条数 | 20 条抽检，粒度 5% |
| **平均区分度** | Σ discrimination / 样本数 | 核心指标 |
| **有效用例比例** | discrimination ≥ 0.3 的条数 / 样本数 | 比平均区分度更抗极端值 |

### 9.2 评测指标（通用）

| 指标 | 公式 | 易错点 |
|---|---|---|
| 请求成功率 | `request_ok` 数 / 总样本数 | — |
| 任务成功率 | `task_ok` 数 / **总样本数** | 分母剔掉失败请求就是作弊 |
| 格式合规率 | JSON 可解析数 / **要求 JSON 的样本数** | 分母不是全部 |
| 平均延迟 | 成功请求耗时之和 / 成功请求数 | 失败耗时不能进分子 |
| P50 / P95 | 成功请求耗时升序，nearest-rank | 用不用插值写死并注释 |
| 吞吐 | 完成样本数 / 总墙钟时间 | 分母含失败样本 |
| 成本 | 输入token×单价 + 输出token×单价 | 标 `source: api/estimated`，单价记日期 |

### 9.3 矩阵对比（最终验证 + 实际产出）

**三个被测对象**（不只是两个 prompt）：

| 编号 | 对象 | 用途 |
|---|---|---|
| A | `prompts/v1.txt` + DeepSeek | 弱基线（区分度的弱参考系） |
| B | `prompts/v2.txt` + DeepSeek | 强基线（区分度的强参考系） |
| C | 另一个模型（通义 / gpt-4o-mini）+ v2 | 证明"这套集子能用来比模型" |

**两个数据集**：人工集 60 条、自动集（过滤后约 120–150 条）。

跑成 3×2 矩阵：

```
                 人工集      自动集
A (v1)            ?%          ?%
B (v2)            ?%          ?%
C (模型B)         ?%          ?%
```

**两个结论必须都给**：

1. **方法论结论**：两集给出的排序是否一致？
   | 情况 | 判定 |
   |---|---|
   | 两集排序相同 | 自动集可替代人工集 |
   | 排序相反 | **自动集不可信**，说明原因 |
   | 一集有差异、一集没差异 | 看哪个集合的区分度更高；区分度低的那个没测出来 |

2. **实际产出结论**：在这套集子上，A / B / C 谁强谁弱、差几个百分点、成本差多少。

**只报第一个结论是不够的**——那是"我造了尺子但没用它量东西"。
**只报第二个也不够**——那是"我拿一把没验证过的尺子量了东西"。
**两个一起，才是完整的评测工作。**

### 9.4 推理性能指标（模块 3，D15–D16）

**必须用流式（`stream=true`）采集**，否则测不出首 token 延迟。

| 指标 | 定义 | 为什么重要 |
|---|---|---|
| **TTFT** | Time To First Token：从发请求到收到第一个 token 的时间 | 用户感知的"卡不卡"，交互式应用最关键 |
| **TPOT** | Time Per Output Token = (总耗时 − TTFT) / (输出 token 数 − 1) | 生成速度，决定长回答要等多久 |
| **端到端延迟** | 请求发出到最后一个 token | 整体体验 |
| **吞吐** | 输出 token 总数 / 总墙钟时间（tokens/s） | 批量跑评测时的效率 |
| **P50 / P95** | 上述指标的分布，不只是均值 | 长尾才是用户骂人的时刻 |
| **错误率** | 失败数 / 总请求，**429 单独统计** | 限流是并发策略问题，不是网络问题 |
| **单位成本** | 总花费 / 成功样本数 | 回答"这活儿多少钱能干完" |

**并发扫描**：1 / 2 / 4 / 8 / 16 各跑一轮，画「并发 vs 吞吐 vs P95」双轴图，找饱和点。

**必须写进报告的三个坑**：
1. **客户端并发度 ≠ 服务端在途请求数**：你设 8 个协程，不代表服务端同时处理 8 个
2. **TTFT 受网络影响极大**：同城/跨城差几十毫秒，报告里要写清测试环境
3. **限流会让 P95 突然爆掉**：429 要单独计数，混进失败里就看不见真因

**质量-性能-成本三维对比**（D16 的核心产出）：
把三个被测对象放在同一张表里，回答"哪个最划算"：

| 对象 | 任务成功率 | P95 延迟 | TTFT | 单样本成本 |
|---|---|---|---|---|
| A | ?% | ?ms | ?ms | ?元 |
| B | ?% | ?ms | ?ms | ?元 |
| C | ?% | ?ms | ?ms | ?元 |

**结论不是"谁最好"，是"谁在什么场景下最划算"**——比如"B 质量最高但成本 3 倍，交互场景用 A 够，批量跑用 B"。

---

## 十、模块划分

```
evalkit/
  spec.py          spec 解析与校验
  generator/
    base.py        生成器接口（策略可插拔）
    single.py      S1 单轮
    batched.py     S2 分批 + 去重
    iterative.py   S3 反馈迭代
    dedup.py       字符级 Jaccard 去重
  dataset.py       加载 + 校验 + hash
  audit.py         可执行性 / 重复 / 标签分布
  ambiguity.py     可判定性检测（3 次一致性）
  discrimination.py 区分度打分
  client.py        ModelClient 接口 + DeepSeekClient + FakeClient
  graders/
    normalize.py / exact.py / json_check.py / field.py / contains.py
  metrics.py       纯函数，不碰 IO
  runner.py        串行 / 并发 + 预热
  diff.py          两次 run 按 sample_id 对齐
  gate.py          门禁判定
  report.py        Markdown
  render_html.py   静态 HTML（matplotlib → base64）
  cli.py           Typer 入口
prompts/
  v1.txt           弱 prompt
  v2.txt           强 prompt（区分度的参考系）
specs/
  extraction.yaml  任务规格
web/               （可选）薄服务
docs/
  annotation-guide.md  人工抽检指南
```

**六条硬性设计约束**：
1. `metrics.py` 纯函数，不读文件不发请求
2. `ModelClient` 是接口，能换模型、能离线测
3. **`generator/` 是可插拔策略**，加第 4 种策略不改其他三个
4. **`discrimination.py` 必须 repeat ≥ 3**，单次结果不算
5. **`source` 字段贯穿全流程**，任何统计都要能按 human/generated 分组
6. `diff.py` / `gate.py` 只读已有结果，不重跑

---

## 十一、明确不做

- 多用户 / 登录 / 权限
- WebSocket / SSE
- Celery / Redis / 消息队列
- React / Vue / 前端构建链
- 在线编辑数据集、在线改评分规则
- 模型训练 / 微调
- 分布式部署

---

## 十二、已知风险（提前写下来）

1. **生成质量可能很差**（可执行率低、正确率低）——这也是结论，报告要如实写"自动生成在当前 spec 下不达标"
2. **区分度可能测不出来**：如果 v1 和 v2 本身差异太小，所有用例的区分度都会接近 0。这时要换更强的对照（比如故意做一个很烂的 prompt）
3. **20 条抽检的粒度是 5%**，只能说明趋势
4. **Jaccard 去重对语义重复无效**：换个说法但意思一样的用例会漏掉
5. **成本翻倍**：区分度要跑 2 个 prompt × 3 轮，生成还要额外调用

---

## 十三、技术栈

```
运行时：python 3.11 / httpx / pydantic / typer / rich / python-dotenv
       pyyaml（spec）/ matplotlib（出图）
服务：  fastapi / uvicorn（可选薄服务）
开发：  pytest / pytest-asyncio / pytest-cov / ruff
工程：  Docker / GitHub Actions
```

**不装**：pandas（数据量小）、SQLAlchemy（一张表手写 SQL）、celery/redis（单机够）、jinja2（拼字符串）

**简历一行**：
```
Python 3.11（asyncio / httpx / Pydantic / Typer）、pytest、SQLite / SQL、
Docker、GitHub Actions、matplotlib
```
