"""每次训练使用独立 Python 进程；串行运行，隔离内存和计时。"""
from datetime import datetime, timezone
import csv
import importlib.metadata
import platform
import random
import statistics
import subprocess
import sys
import time

import psutil

from .common import ROOT, config, read_json, write_json, sentences, sha256
from .modeling import configurations, evaluate, load_model


def run_experiments():
    cfg = config()
    if not (ROOT / "results/data_summary.json").exists():
        raise FileNotFoundError("先运行 python -m ngram_lab.cli prepare")
    logs = ROOT / "results/logs"
    logs.mkdir(parents=True, exist_ok=True)
    packages = {name: importlib.metadata.version(name) for name in ["nltk", "numpy", "matplotlib", "psutil", "pytest"]}
    env = {"system": platform.platform(), "python": sys.version, "cpu": platform.processor(),
           "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
           "ram_gib": psutil.virtual_memory().total / 2**30, "available_ram_gib_at_start": psutil.virtual_memory().available / 2**30,
           "gpu_used": False, "parallel_training_jobs": 1, "packages": packages,
           "started_utc": datetime.now(timezone.utc).isoformat(), "config": cfg,
           "config_sha256": sha256(ROOT / "config.json"),
           "data_summary_sha256": sha256(ROOT / "results/data_summary.json")}
    write_json(ROOT / "results/environment.json", env)
    tasks = [(s["id"], repeat) for s in configurations() for repeat in range(1, cfg["repeats"] + 1)]
    random.Random(cfg["seed"]).shuffle(tasks)
    write_json(ROOT / "results/run_order.json", tasks)
    started = time.perf_counter()
    for number, (model_id, repeat) in enumerate(tasks, 1):
        print(f"[{number}/{len(tasks)}] {model_id}, repeat={repeat}", flush=True)
        command = [sys.executable, "-m", "ngram_lab.cli", "worker", "--model", model_id, "--repeat", str(repeat)]
        with (logs / f"{model_id}_r{repeat}.log").open("w", encoding="utf-8") as output:
            process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
            child = psutil.Process(process.pid)
            job_start = time.perf_counter()
            while process.poll() is None:
                try:
                    rss = child.memory_info().rss / 2**20
                except psutil.NoSuchProcess:
                    break
                if rss > cfg["training_memory_limit_mb"] or time.perf_counter() - job_start > cfg["training_timeout_seconds"]:
                    process.kill()
                    process.wait()
                    raise RuntimeError(f"{model_id} 超过内存/时间预算，已停止该实验；查看日志")
                time.sleep(0.2)
            if process.wait() != 0:
                raise RuntimeError(f"{model_id} 训练失败，查看 results/logs/{model_id}_r{repeat}.log")
    summaries = []
    for spec in configurations():
        rows = [read_json(ROOT / "results/runs" / f"{spec['id']}_r{r}.json") for r in range(1, cfg["repeats"] + 1)]
        first = rows[0]
        times = [r["train_seconds"] for r in rows]
        summaries.append({**spec, "train_tokens": first["train_tokens"], "train_sentences": first["train_sentences"],
                          "train_articles": first["train_articles"], "train_seconds_median": statistics.median(times),
                          "train_seconds_min": min(times), "train_seconds_max": max(times),
                          "peak_rss_mb_max": max(r["peak_rss_mb"] for r in rows),
                          "incremental_rss_mb_max": max(r["incremental_rss_mb"] for r in rows),
                          "model_mb": first["model_mb"], "distinct_ngrams": first["distinct_ngrams"],
                          "dev": first["dev"], "save_seconds": first["save_seconds"]})
    # 先冻结验证集选择结果，再读取测试集，测试集不参与参数选择。
    candidates = [s for s in summaries if s["fraction"] == 1 and s["dev"]["ppl"] is not None]
    winner = min(candidates, key=lambda s: s["dev"]["ppl"])
    wb_winner = min((s for s in candidates if s["smoothing"] == "witten_bell"), key=lambda s: s["dev"]["ppl"])
    write_json(ROOT / "results/selection.json", {"criterion": "minimum dev perplexity among full-data models",
               "best_id": winner["id"], "dev_ppl": winner["dev"]["ppl"], "best_full_data_wb_id": wb_winner["id"],
               "frozen_before_test_utc": datetime.now(timezone.utc).isoformat()})
    test = sentences("test")
    for summary in summaries:
        print(f"test evaluation: {summary['id']}", flush=True)
        model = load_model(summary["id"])
        summary["test"] = evaluate(model, test)
        del model
    write_json(ROOT / "results/summary.json", summaries)
    with (ROOT / "results/metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        columns = ["id", "order", "smoothing", "gamma", "fraction", "train_tokens", "train_seconds_median", "train_seconds_min", "train_seconds_max", "peak_rss_mb_max", "model_mb", "dev_ppl", "test_ppl", "test_zero_events"]
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        for s in summaries:
            row = {k: s[k] for k in columns if k in s}
            row.update(dev_ppl=s["dev"]["ppl"] if s["dev"]["ppl"] is not None else "inf",
                       test_ppl=s["test"]["ppl"] if s["test"]["ppl"] is not None else "inf",
                       test_zero_events=s["test"]["zero_events"])
            writer.writerow(row)
    env["experiments_wall_seconds"] = time.perf_counter() - started
    env["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(ROOT / "results/environment.json", env)
    print(f"complete: selected {winner['id']} using dev only", flush=True)
