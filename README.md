# Research Paper Assistant Agent

## 项目简介
一个面向论文场景的轻量 Agent：支持**总结（summary）**、**批判（critique）**、**对比（compare）**三类任务，并具备文档记忆、对话记忆和自动评测能力。

## 核心改进
### 1. Planner：从关键词规则到 LLM 意图识别
- 问题：仅靠关键词命中做路由，复杂输入下容易误判工具。
- 改动：接入 LLM 进行工具选择，限定输出 `summary | compare | critique` 的结构化结果；同时保留 fallback 规则，提升稳健性。

### 2. DocumentMemory：支持双论文与来源可追踪检索
- 问题：原实现偏单文档，compare 场景无法稳定区分片段来源。
- 改动：
  - `load()` 支持加载两篇论文（`paper_a` / `paper_b`）。
  - `top_k(query, source=...)` 支持按单篇或双篇检索。
  - `top_k` 返回 `RetrievedChunk`（包含 `source` + `chunk`），避免来源信息丢失。

### 3. Tools：从固定模板到 LLM 驱动
- 问题：固定模板输出僵硬，信息利用率低。
- 改动：`summarize.py`、`critique.py`、`compare.py` 已改为 LLM 驱动：
  - 先检索 chunk 证据，再调用 LLM。
  - 强约束 JSON 输出并做解析/归一化。
  - 解析失败或 API 异常时自动兜底到本地 fallback，保证可用性。
- 额外稳定性改动：
  - compare 在只有一篇论文时直接返回明确提示，不再伪造“两篇对比”的展示。

### 4. Evaluator：跳过不适用 case，避免失真评分
- 问题：单论文输入时若强行评估 compare，会拉低总体分数且不公平。
- 改动：`eval/benchmark.json` 为 compare case 增加 `requires_two_papers: true`；`eval/evaluator.py` 在仅有一篇论文时自动跳过该 case，且**不计入平均分**。

## 运行方式
```bash
pip install -r requirements.txt
python main.py --paper data/sample_paper.txt --query "请总结这篇论文的核心贡献"
python scripts/run_eval.py
```

## 使用说明
- 当前 CLI 入口 `main.py` 默认接收单论文参数 `--paper`。
- 双论文对比能力已在 `DocumentMemory` 与 `compare` 工具层实现；若要从 CLI 直接做双论文 compare，可后续扩展为 `--paper-a` / `--paper-b` 参数。

## 环境变量
如需启用 LLM 路径，请配置：
- export DASHSCOPE_API_KEY=""
- export DASHSCOPE_BASE_URL=""

## 项目结构（简化）
```text
rebuild_gpt_agent/
├── main.py
├── config.py
├── agent/
│   ├── app.py
│   ├── planner.py
│   └── memory.py
├── tools/
│   ├── summarize.py
│   ├── critique.py
│   ├── compare.py
│   └── parser.py
├── eval/
│   ├── benchmark.json
│   └── evaluator.py
├── data/
│   └── sample_paper.txt
└── scripts/
    └── run_eval.py
```
