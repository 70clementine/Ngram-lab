"""核对已保存的真实结果与数据、模型、报告是否一致；不重新训练。"""
from pathlib import Path
import json
import re
import statistics
import subprocess
import sys

from ngram_lab.common import ROOT, read_json, sha256
from ngram_lab.modeling import configurations


def main():
    summary = read_json(ROOT / "results/summary.json")
    data = read_json(ROOT / "results/data_summary.json")
    assert {s["id"] for s in summary} == {s["id"] for s in configurations()}
    assert data["sampled_tokens"] == sum(s["tokens"] for s in data["splits"].values())
    for split, stats in data["splits"].items():
        assert sha256(ROOT / f"data/processed/{split}.txt") == stats["file_sha256"]
    for model in summary:
        runs = [read_json(p) for p in sorted((ROOT / "results/runs").glob(f"{model['id']}_r*.json"))]
        assert len(runs) == data["config"]["repeats"]
        assert model["train_seconds_median"] == statistics.median(r["train_seconds"] for r in runs)
        assert (ROOT / f"models/{model['id']}.pkl").exists()
        assert model["test"]["events"] == data["splits"]["test"]["tokens"] + data["splits"]["test"]["sentences"]
        assert (model["test"]["ppl"] is None) == bool(model["test"]["zero_events"])
    selection = read_json(ROOT / "results/selection.json")
    eligible = [s for s in summary if s["fraction"] == 1 and s["dev"]["ppl"] is not None]
    assert selection["best_id"] == min(eligible, key=lambda x:x["dev"]["ppl"])["id"]
    report = (ROOT / "实验报告.md").read_text(encoding="utf-8")
    assert not any(ord(c) < 32 and c not in "\n\r\t" for c in report)
    for path in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report):
        assert (ROOT / path).is_file(), path
    for filename in ["data/raw/people_daily_1998.txt", "data/processed/train.txt", "models/wb_n3.pkl", ".vscode/settings.json"]:
        assert subprocess.run(["git", "check-ignore", "--quiet", filename], cwd=ROOT).returncode == 0
    generations = read_json(ROOT / "results/generations.json")
    assert len(generations) == 8 * len(data["config"]["generation_seeds"]) + 1
    for row in generations:
        assert row["length"] == len(row["generated_tokens"])
        assert not {"<s>", "</s>", "<UNK>"}.intersection(row["generated_tokens"])
    print("PASS: result consistency, split hashes, evaluation denominator, validation selection, report images and Git exclusions")


if __name__ == "__main__":
    main()
