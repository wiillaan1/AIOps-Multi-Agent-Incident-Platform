# AIOps Agent Platform

多 Agent 运维故障定位与自愈系统。

## 功能

系统接收一条告警后，由 Monitor、RCA、Heal、Change 四个 Agent 在 LangGraph 编排下依次完成异常确认、根因分析、修复方案生成和变更审批，每一步都写入故障时间线和审计日志。后端是 FastAPI，前端是 React，提供故障列表、故障详情、服务拓扑、Agent 状态和评测结果等页面。配置 LLM 后会调用 LLM 做结构化复核；不配置时自动降级为规则引擎，完整流程照样能跑通。

- 根因定位：Monitor Agent 用 Isolation Forest 等多种算法投票确认异常，RCA Agent 结合服务拓扑遍历、贝叶斯推理、知识库和历史记忆，给出根因、置信度和影响链。
- 分级自愈：Heal Agent 按爆炸半径把修复动作分为 L0（自动执行）、L1（需 oncall 确认）、L2（需 TL 审批）三级，生成对应的 kubectl 命令，但只做 dry-run 模拟，不会真正执行；处于发布窗口内时强制 dry-run。
- 变更审批：Change Agent 根据变更类型、时间段等因素计算风险分，据此决定自动通过、仅通知、升级人工、回滚或延后，超时未处理会自动升级。
- 记忆与自进化：记忆分短期、工作、长期三层，长期记忆存在 ChromaDB 中，RCA 结论入库后供下一次分析检索相似的历史故障；自进化循环会收集 badcase、分析失败模式、生成 prompt 修改并做 A/B 对比，最后由人工决定 promote 或 reject，不会自动生效。
- 自动评测：在 dev / test / holdout 三份评测集上，按根因匹配、置信度、证据引用、无幻觉、语义匹配五个维度打分，结果写入历史库，支持与 baseline 对比，并附带 GitHub Actions 评测门禁。

数据说明：系统使用的全部是模拟数据。服务拓扑是 8 个服务的模拟微服务拓扑（api-gateway、order-service、payment-service、inventory-service、user-service、mysql-primary、redis-cache、elasticsearch），故障场景是内置的故障注入数据集（`backend/app/data/datasets.py`）。评测集（`data/badcases/`）参照 AIOps Challenge 2020、Microsoft AIOpsLab、Loghub HDFS、SMD 的故障类型静态生成，不包含这些数据集的原始数据。

## 启动

以下命令都从仓库根目录开始执行。

### 1. 环境要求

- Python 3.13
- Node.js 26 和 npm 11（只有启动前端时才需要）
- 不需要 Docker、LLM API key、Prometheus 或 Loki，缺少这些时系统会自动降级运行

### 2. 安装依赖

后端：

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

前端（可选）：

```bash
cd frontend
npm ci
```

### 3. 配置环境变量

```bash
cd backend
cp .env.example .env
```

所有变量都有默认值，不改 `.env` 也能以规则引擎模式启动。`.env` 按当前工作目录加载，所以后端必须在 `backend/` 目录下启动。各变量的用途如下，完整说明见 `backend/.env.example`。

| 变量 | 用途 |
|---|---|
| `APP_ENV` | 运行环境，取值 development / testing / staging / production；非 development 时关闭 `/docs` 和演示接口 |
| `APP_HOST`、`APP_PORT` | 监听地址和端口 |
| `APP_LOG_LEVEL`、`APP_DEBUG` | 日志级别、调试模式 |
| `APP_CORS_ORIGINS` | 允许跨域访问的前端地址，JSON 数组格式 |
| `LLM_PROVIDER` | LLM 提供商，取值 openai / deepseek / ollama / mimimax / anthropic；anthropic 需额外安装 `langchain-anthropic` |
| `LLM_API_KEY` | LLM 密钥，留空则以规则引擎模式运行 |
| `LLM_BASE_URL` | 自定义的 OpenAI 兼容端点，mimimax 和 ollama 必填 |
| `LLM_MODEL`、`LLM_TEMPERATURE`、`LLM_MAX_TOKENS` | 模型名、采样温度、最大生成 token 数 |
| `LLM_TIMEOUT_SECONDS`、`LLM_MAX_RETRIES` | 单次请求超时时间、重试次数 |
| `LLM_INPUT_COST_PER_1K`、`LLM_OUTPUT_COST_PER_1K` | token 单价，用于统计成本 |
| `LANGFUSE_ENABLED`、`LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_HOST` | 是否把 LLM 调用上报到 Langfuse，以及对应的密钥和地址 |
| `LANGFUSE_RELEASE`、`LANGFUSE_ENVIRONMENT` | 上报时附带的版本号和环境名 |
| `AGENT_RCA_MIN_CONFIDENCE` | RCA 结论被判定为高置信的最低置信度 |
| `AGENT_RCA_V2_ENABLED` | 是否启用 RCA 合成优化（相似度修正、冲突降置信、未验证记忆不参与加成） |
| `AGENT_VERIFICATION_ENABLED` | 是否启用调查验证闭环（候选验证、反证、重新规划） |
| `MEMORY_SHORT_TERM_MAX_ITEMS`、`MEMORY_SIMILARITY_THRESHOLD`、`MEMORY_RETENTION_DAYS` | 短期记忆容量、相似度阈值、记忆保留天数 |
| `WS_HEARTBEAT_INTERVAL_SECONDS` | WebSocket 心跳间隔 |

`PROMETHEUS_URL`、`LOKI_URL`、`AIOPS_FORCE_FALLBACK` 和 `AIOPS_USE_*` 这几个开关是从进程环境变量读取的，写进 `.env` 不会生效，需要加在启动命令前面，例如 `PROMETHEUS_URL=http://localhost:9090 uvicorn app.main:app`。具体说明见 `.env.example` 末尾。

### 4. 启动

后端（在 `backend/` 目录、已激活 venv 的终端中执行）：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

日志中出现 `Startup complete` 和 `Uvicorn running on http://0.0.0.0:8000` 就表示启动成功。打开另一个终端检查：

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

`/health` 返回 `"status":"healthy"`；`/ready` 中 chromadb、memory、knowledge_base、langgraph 都应为 `ready`。本地没跑 Prometheus 时 prometheus 显示 `unreachable`，这是正常的。接口文档在 http://localhost:8000/docs 。

前端（可选，在 `frontend/` 目录执行）：

```bash
npm run dev
```

浏览器打开 http://localhost:3000 ，`/api` 和 `/ws` 请求会被代理到 `localhost:8000`。

### 5. 触发一次示例故障

模拟 order-service CPU 使用率达到 95%（阈值 80%）的告警：

```bash
curl -s -X POST http://localhost:8000/api/v1/incidents/trigger \
  -H "Content-Type: application/json" \
  -d '{"service": "order-service", "metric": "cpu_usage_percent", "value": 95, "threshold": 80, "severity": "critical"}'
```

接口返回 `"status":"accepted"` 和一个 `incident_id`。等几秒后，把下面命令中的 `<incident_id>` 换成上一步返回的值，查看处理结果：

```bash
curl -s http://localhost:8000/api/v1/incidents/<incident_id> | python3 -c "import sys, json; d = json.load(sys.stdin); print('状态:', d['state']); print('根因:', d['rca']['root_cause'], round(d['rca']['confidence'], 2)); print('自愈:', d['heal']); print('变更:', d['change']); [print(' ', t['phase'], t['actor'], t['action']) for t in d['timeline']]"
```

跑通时状态为 `resolved`，能看到 RCA 给出的根因和置信度、Heal 判定的自愈级别和 dry-run 状态、Change 给出的风险分和审批状态。时间线依次经过 detection、triage、rca、mitigation、resolution 五个阶段。同一故障生成的 Runbook 可以通过 `GET /api/v1/incidents/<incident_id>/runbook` 查看。

同一个服务、指标、严重级别的告警在 5 分钟内重复提交会返回 `"status":"deduplicated"`。想再触发一次，换一个 `service` 或 `severity` 即可。

### 6. 运行测试

在 `backend/` 目录、已激活 venv 的终端中执行：

```bash
python -m pytest tests/ -q
python -m pytest app/tests -q
```

测试会自动禁用真实 LLM 调用，不需要 API key。

运行评测并查看历史记录：

```bash
AIOPS_FORCE_FALLBACK=1 python -m app.eval run --split dev
python -m app.eval history
```

未配置 LLM 时 RCA 走规则引擎。评测能完整跑完，但规则引擎覆盖不了评测集中的大部分根因类型，所以通过率会很低。

运行一轮自进化循环，然后查看结果：

```bash
AIOPS_FORCE_FALLBACK=1 python -m app.evolution run --badcases bc-aiops2020-020,bc-aiops2020-001 --prompt "现有 RCA prompt"
python -m app.evolution status
```

循环结束后停在待人工 review 状态，需要手动决定通过或拒绝：

```bash
python -m app.evolution promote <cycle_id> --reviewer <审核人>
python -m app.evolution reject <cycle_id> --reviewer <审核人>
```
