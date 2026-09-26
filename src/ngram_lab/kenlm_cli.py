"""KenLM 主入口；训练与生成在 Docker 内，作图也可以本机执行。"""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="KenLM 人民日报实验")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["prepare", "experiment", "generations", "report", "all"]:
        sub.add_parser(name)
    gen = sub.add_parser("generate")
    gen.add_argument("--model", default=None)
    gen.add_argument("--prompt", help="空格分词；默认使用作业给定开头")
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--top-k", type=int, default=20)
    gen.add_argument("--seed", type=int, default=11)
    gen.add_argument("--max-tokens", type=int, default=60)
    gen.add_argument("--greedy", action="store_true")
    args = parser.parse_args()
    from . import kenlm_pipeline as p
    if args.command in ["prepare", "all"]:
        p.prepare_inputs()
    if args.command in ["experiment", "all"]:
        p.run_experiments()
    if args.command in ["generations", "all"]:
        p.run_generations()
    if args.command in ["report", "all"]:
        from .kenlm_reporting import build_report
        build_report()
    if args.command == "generate":
        model = args.model or p.read_json(p.OUT / "selection.json")["selected"]
        result = p.generate(model, args.prompt.split() if args.prompt else None,
                            args.temperature, args.top_k, args.seed, args.max_tokens, args.greedy)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
