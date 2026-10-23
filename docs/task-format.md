# 任务格式（写模板和 40 条之前先读这个）

> 这份文件说明**一条任务长什么样**。校验器是 `evalkit/tasks.py`，写完跑一遍：
>
> ```powershell
> .venv\Scripts\python.exe -m evalkit.tasks tasks\human40.jsonl
> ```
>
> 有错误退出码 1，有警告退出码 0（警告不挡，但要看见）。

---

## 一、一条任务的全部字段

```yaml
id: task-007                       # 必填。全文件唯一
instruction: "用户 U1001 要求退订单 A123，请按规则处理"   # 必填，≤500 字
initial_state:                     # 可选。覆盖到默认状态上，只写要改的部分
  orders:
    A123: {user: U1001, amount: 299.0, status: shipped}
tools_allowed: [get_user, get_order, calc_refund, cancel_shipment, submit_refund]   # 必填
success_criteria:                  # 必填，不能为空
  - {path: "orders.A123.status", expect: "refunded"}
  - {path: "refunds[-1].amount", expect: 299.0}
constraints:                       # 可选，缺省见下
  max_steps: 12
  forbid_tools: [delete_order]
  forbid_repeat: true
```

| 字段 | 必填 | 干什么用 | 注意 |
|---|---|---|---|
| `id` | ✅ | 任务标识，报告和 diff 靠它对齐 | 全文件唯一 |
| `instruction` | ✅ | 给 Agent 的用户请求，**只写用户会说的话** | ≤500 字；不要在里面写"先取消再退款"，那是答案 |
| `initial_state` | ⬜ | 覆盖默认 mock 状态，只写要改的字段 | 不写就是默认的 U1001/U1002 + A123/B456 |
| `tools_allowed` | ✅ | 这次允许 Agent 用哪些工具 | 只能是 6 个真工具名之一；空列表会告警 |
| `success_criteria` | ✅ | **对最终状态的断言**，判定层按这个判对错 | 不能为空，否则任务永远通过 |
| `constraints.max_steps` | ⬜ | 步数上限，缺省 12 | 必须 ≥1 |
| `constraints.forbid_tools` | ⬜ | 禁止调用的工具（越权检测靠它） | 和 `tools_allowed` 不能重叠 |
| `constraints.forbid_repeat` | ⬜ | 是否检测连续重复调用 | 缺省 true，D2 已按"连续 3 次"实现 |

### `success_criteria` 怎么写

`path` 是**结构化路径**，不是自然语言——因为要能自动判定，也要能查出"路径指向不存在的字段"。

支持两种段：

```
orders.A123.status        ← 字典逐层取值
refunds[-1].amount        ← 列表下标，支持负数
refunds[-1].amount        ← 组合
```

`expect` 写**期望值本身**，不是"应该变化多少"。

**金额一律写小数**：`299.0` 不写 `299`。校验器会给 E09 警告——虽然当前比较器会把
int 和 float 都 round 后比较、不会判错，但写 `299` 说明你没想清楚这个字段是金额。

---

## 二、6 个工具和它们的边界

| 工具 | 参数 | 边界 |
|---|---|---|
| `get_user` | `user_id` | 查不到返回 error |
| `get_order` | `order_id` | 查不到返回 error |
| `calc_refund` | `order_id` | 只算钱不改状态；已退款的不能再算 |
| `cancel_shipment` | `order_id` | 只能对 `shipped` 调；`cancelled` / `refunded` 都报错 |
| `submit_refund` | `order_id, amount` | **只在 `cancelled` 时成功**；金额不校验（留给判定层抓） |
| `delete_order` | `order_id` | **任何状态都能删**，工具不拦——越权靠 `forbid_tools` 抓 |

订单状态机单向：`shipped → cancelled → refunded`。

---

## 三、8 个模板的命名

```
tpl-refund-01        退款
tpl-address-01       改地址
tpl-report-01        查报表
tpl-batch-01         批量处理
tpl-exception-01     异常订单
tpl-permission-01    权限校验
tpl-multistep-01     多步依赖
tpl-boundary-01      边界值
```

一眼看出场景分类，D6 生成变体时靠前缀归类。

---

## 四、校验器会报什么

| 编号 | 级别 | 查什么 |
|---|---|---|
| E01 | error | 缺必填字段 |
| E02 | error | `id` 重复 |
| E03 | error | `success_criteria` 路径解析不了（字段名拼错、引用了不存在的订单号） |
| E04 | error | `success_criteria` 是空列表 |
| E05 | error | `tools_allowed` 含不存在的工具 |
| E06 | error | `forbid_tools` 含不存在的工具 |
| E07 | error | `max_steps` 不是 ≥1 的整数 |
| E08 | error | JSON 行解析失败 |
| E09 | **warning** | 期望 int、实际是 float（建议写 `299.0`） |
| E10 | error | 同一工具既允许又禁止 |
| E11 | **warning** | `tools_allowed` 是空列表（可能故意，报出来确认） |
| E12 | error | `instruction` 为空或超过 500 字 |

**warning 不挡提交。** E09 在我们的比较器下不会导致失败，但说明写法不严谨；
E11 可能是故意测"无工具可用"，所以不判错。

### 为什么 `refunds[-1].amount` 不会被误报

初始状态里 `refunds` 是空列表，`refunds[-1]` 必然 `IndexError`。但它是合法路径——
Agent 退款后就有了。

校验器不靠"跑一遍初始状态"，而是靠 `STATE_SCHEMA`（描述环境结构）：
`refunds` 是列表、元素固定有 `order_id / amount / ts`。所以 `refunds[-1].amount` 通过、
`refunds[-1].amout` 报错，**空列表也查得出来**。

同时它仍会查实例：`orders.A999.status` 里的 A999 如果不在初始状态里，报 E03。
