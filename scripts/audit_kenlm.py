"""从已训练模型独立复算结果，不重新训练；将断言结果保存以便查验。"""
import re
import statistics
import subprocess
from datetime import datetime, timezone

import kenlm
from ngram_lab.common import ROOT, read_json, write_json, sha256
from ngram_lab.kenlm_pipeline import OUT, DATA, MODELS, evaluate, settings, RARE


def main():
    rows = read_json(OUT / "summary.json")
    cfg = settings()
    assert {s["id"] for s in rows} == {s["id"] for s in cfg["models"]}
    data = read_json(OUT / "data.json")
    for split, metadata in data["splits"].items():
        assert sha256(DATA / f"{split}.txt") == metadata["sha256"]
    for s in rows:
        runs = [read_json(OUT / f"runs/{s['id']}_r{r}.json") for r in range(1, cfg["repeats"] + 1)]
        assert statistics.median(r["train"]["elapsed_seconds"] for r in runs) == s["train_seconds_median"]
        assert sha256(MODELS / f"{s['id']}.arpa") == runs[-1]["arpa_sha256"]
        assert all(r["train"]["exit_code"] == 0 and r["build"]["exit_code"] == 0 for r in runs)
        assert (MODELS / f"{s['id']}.bin").stat().st_size / 2**20 == s["binary_mib"]
        model = kenlm.Model(str(MODELS / f"{s['id']}.bin"))
        for split in ["dev", "test"]:
            actual = evaluate(model, DATA / f"{split}.txt")
            assert abs(actual["ppl"] - s[split]["ppl"]) < 1e-8
            assert actual["events"] == data["splits"][split]["tokens"] + data["splits"][split]["sentences"]
            assert actual["model_oov_events"] == 0
    winner = min([s for s in rows if "_mem" not in s["id"]], key=lambda s: (s["dev"]["ppl"], s["binary_mib"]))
    assert read_json(OUT / "selection.json")["selected"] == winner["id"]
    samples = read_json(OUT / "generations.json")
    assert len(samples) == 5 * len(cfg["generation_seeds"]) + 1 + 8
    for s in samples:
        assert s["length"] == len(s["generated_tokens"])
        assert not set(s["generated_tokens"]) & {RARE, "<s>", "</s>", "<unk>"}
    report = (ROOT / "实验报告.md").read_text(encoding="utf-8")
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report)
    assert len(images) == 6
    for image in images:
        assert (ROOT / image).is_file()
    # Linux 容器通常没有 git，忽略规则由宿主机另行检查。
    test = subprocess.run(["python", "-m", "pytest", "-q"], cwd=ROOT, text=True, capture_output=True)
    assert test.returncode == 0, test.stdout + test.stderr
    result = {"utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
              "models_recomputed": len(rows), "generation_samples": len(samples),
              "figures_checked": len(images), "pytest_output": test.stdout.strip(),
              "checks": ["输入哈希", "计时汇总", "ARPA 哈希", "模型体积", "16组评估重算", "PPL 分母", "验证选型", "生成符号", "报告图片"]}
    write_json(OUT / "verification.json", result)
    print(test.stdout)
    print("KenLM 实测结果审计通过")


if __name__ == "__main__":
    main()
