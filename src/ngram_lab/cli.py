"""入口：python -m ngram_lab.cli --help。"""
import argparse
import json

from .common import config


def main():
    parser = argparse.ArgumentParser(description="人民日报 n-gram：下载、清洗、实验、生成与报告")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["prepare", "experiment", "generations", "report", "all"]:
        sub.add_parser(name)
    worker = sub.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--model", required=True)
    worker.add_argument("--repeat", type=int, required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--model", default="wb_n3")
    gen.add_argument("--prompt", help="按空格分词的开头；省略时使用作业开头")
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--top-k", type=int, default=20)
    gen.add_argument("--seed", type=int, default=11)
    gen.add_argument("--max-tokens", type=int, default=60)
    gen.add_argument("--greedy", action="store_true")
    args = parser.parse_args()
    if args.command in ["prepare", "all"]:
        from .data import prepare
        prepare()
    if args.command in ["experiment", "all"]:
        from .experiment import run_experiments
        run_experiments()
    if args.command in ["generations", "all"]:
        from .generation import run_generation
        run_generation()
    if args.command in ["report", "all"]:
        from .reporting import build_report
        build_report()
    if args.command == "worker":
        from .modeling import train_worker
        train_worker(args.model, args.repeat)
    if args.command == "generate":
        from .modeling import configurations, load_model
        from .generation import generate
        spec = next(s for s in configurations() if s["id"] == args.model)
        prompt = args.prompt.split() if args.prompt else config()["prompt_tokens"]
        result = generate(load_model(args.model), spec, prompt, args.seed, args.temperature, args.top_k, args.greedy, args.max_tokens)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
