from __future__ import annotations

import argparse

from agent.app import ResearchPaperAgent
from tools.parser import load_text_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research Paper Assistant Agent")
    parser.add_argument("--paper", required=True, help="论文 txt 文件路径")
    parser.add_argument("--query", required=True, help="用户问题")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    text = load_text_file(args.paper)

    agent = ResearchPaperAgent()
    agent.load_document(text)
    output = agent.ask(args.query)

    print("=== Plan ===")
    print(f"Intent: {output.plan_intent}")
    print(f"Steps: {output.plan_steps}")
    print(f"Tool: {output.tool_name}")
    print("\n=== Answer ===")
    print(output.answer)


if __name__ == "__main__":
    main()
