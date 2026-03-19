# Research Paper Assistant Agent (Scaffold)

这是一个从 `Rubuild-GPT` 升级而来的 **AI Agent 项目脚手架**，目标不是继续做一个简单的 LLM demo，而是搭建一个具备以下能力的最小可运行系统：

- Planner：先拆解任务，再执行
- Tools：按任务选择工具，而不是直接胡答
- Memory：保存文档内容与历史交互
- Evaluation：对回答质量做基准测试

## 项目结构

```text
rubuild_gpt_agent_upgrade/
├── main.py                  # CLI 入口
├── config.py                # 配置
├── requirements.txt         # 依赖
├── README.md
├── agent/
│   ├── app.py               # Agent 主流程
│   ├── planner.py           # 任务规划
│   ├── memory.py            # 对话与文档记忆
│   └── llm.py               # LLM 抽象层
├── tools/
│   ├── base.py              # Tool 基类
│   ├── summarize.py         # 摘要工具
│   ├── critique.py          # 论文批判工具
│   ├── compare.py           # 比较工具
│   └── parser.py            # 文档解析与切分
├── eval/
│   ├── benchmark.json       # 测试样例
│   └── evaluator.py         # 自动评估
├── data/
│   └── sample_paper.txt     # 示例文档
└── scripts/
    └── run_eval.py          # 评估入口
```

## 先做什么

1. 把你的论文/项目文本放进 `data/`。
2. 先跑 `main.py`，确认 Planner + Tool + Memory 能跑通。
3. 再跑 `scripts/run_eval.py`，得到一个最基础的 benchmark 分数。
4. 后续你可以继续加：
   - RAG / FAISS
   - 真正的 OpenAI / Qwen function calling
   - Web search
   - Streamlit UI

## 运行

```bash
pip install -r requirements.txt
python main.py --paper data/sample_paper.txt --query "请总结这篇论文的核心贡献"
python scripts/run_eval.py
```

## 这个脚手架故意保持简洁

因为你当前最大的缺口不是“功能多”，而是：

- 有没有 Planner
- 有没有 Tool routing
- 有没有 Memory
- 有没有 Eval

先把结构立起来，再谈复杂化。
