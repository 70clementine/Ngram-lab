"""KenLM 实验：Python 编排，lmplz 计数/平滑，build_binary 编译，绑定评测。

在 Linux Docker 内执行。每个结果都关联命令、语料哈希和原始日志。
"""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import tempfile
import threading
import time

import numpy as np

from .common import ROOT, read_json, write_json, sha256

OUT = ROOT / "results/kenlm"
DATA = ROOT / "data/kenlm"
MODELS = ROOT / "models/kenlm"
# 普通词符号，有真实训练计数；不是 KenLM 内置的 <unk>。
RARE = "〈低频词〉"


def settings():
    return read_json(ROOT / "kenlm_config.json")


def prepare_inputs():
    """复用经过核验的清洗/文章划分，统一低频词表示以保证 PPL 可比。"""
    from .data import prepare
    prepare()
    DATA.mkdir(parents=True, exist_ok=True)
    vocab = read_json(ROOT / "data/processed/vocabulary.json")
    assert RARE not in vocab
    stats = {}
    for name in ["train", "dev", "test"]:
        lines = (ROOT / f"data/processed/{name}.txt").read_text(encoding="utf-8").splitlines()
        sentences = [[RARE if w == "<UNK>" else w for w in line.split()] for line in lines]
        assert not any(w in {"<s>", "</s>", "<unk>", "<UNK>"} for s in sentences for w in s)
        path = DATA / f"{name}.txt"
        path.write_text("\n".join(" ".join(s) for s in sentences) + "\n", encoding="utf-8")
        stats[name] = {"sha256": sha256(path), "sentences": len(sentences),
                       "tokens": sum(map(len, sentences)), "rare_tokens": sum(s.count(RARE) for s in sentences)}
    write_json(DATA / "vocabulary.json", vocab + [RARE])
    write_json(OUT / "data.json", {"splits": stats, "lexical_vocabulary": len(vocab),
                                   "rare_symbol": RARE, "vocabulary_sha256": sha256(DATA / "vocabulary.json")})
    return stats


def run_timed(command, label, timeout):
    """GNU time 记录子进程峰值 RSS；日志时间为 Python 读到该行的时刻。"""
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    resource_file = logs / f"{label}.resource.json"
    log_file = logs / f"{label}.log"
    fmt = '{"wall_seconds":%e,"user_seconds":%U,"system_seconds":%S,"peak_rss_kib":%M,"exit_code":%x}'
    started = time.perf_counter()
    proc = subprocess.Popen(["/usr/bin/time", "-f", fmt, "-o", str(resource_file), *command],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", start_new_session=True)
    events = []
    completion = []

    def drain():
        with log_file.open("w", encoding="utf-8") as f:
            for line in proc.stdout:
                f.write(line)
                f.flush()
                events.append({"seconds": time.perf_counter() - started, "text": line.rstrip()})
        # 在管道 EOF 时计时，避免 wait(timeout) 的轮询周期给短任务加上约 50ms。
        completion.append(time.perf_counter() - started)

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        import os
        import signal
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        thread.join()
        raise TimeoutError(f"{label} 超过 {timeout}s，已终止整个进程组")
    thread.join()
    elapsed = completion[0]
    write_json(logs / f"{label}.events.json", events)
    if code:
        raise RuntimeError(f"{label} 失败: {log_file.read_text(encoding='utf-8')[-2500:]}")
    metrics = read_json(resource_file)
    metrics.update({"elapsed_seconds": elapsed, "command": command,
                    "log": str(log_file.relative_to(ROOT)), "events": str((logs / f"{label}.events.json").relative_to(ROOT))})
    return metrics


def evaluate(model, path):
    """语料级 PPL: 10^(-总 log10 概率 / (词数+句数))，计 EOS、不计 BOS。"""
    total, events, words, oov, rare = 0.0, 0, 0, 0, 0
    matched_orders = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        words += len(tokens)
        rare += tokens.count(RARE)
        for logp, length, is_oov in model.full_scores(line, bos=True, eos=True):
            total += logp
            events += 1
            oov += int(is_oov)
            matched_orders[length] += 1
    return {"ppl": 10 ** (-total / events), "cross_entropy_bits": -total * math.log2(10) / events,
            "log10_probability": total, "events": events, "tokens": words,
            "model_oov_events": oov, "rare_tokens": rare, "rare_rate": rare / words,
            "matched_orders": dict(sorted(matched_orders.items()))}


def arpa_counts(path):
    counts = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            match = re.match(r"ngram (\d+)=(\d+)", line)
            if match:
                counts[int(match[1])] = int(match[2])
            if line.startswith("\\1-grams"):
                break
    return counts


def run_experiments():
    import kenlm
    cfg = settings()
    MODELS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if not (DATA / "train.txt").exists():
        prepare_inputs()
    started = time.perf_counter()
    environment = {"utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
                   "python": platform.python_version(), "kenlm_commit": Path('/opt/kenlm/COMMIT').read_text().strip(),
                   "cpu_limit": cfg["container_cpus"], "memory_limit": cfg["container_memory"],
                   "config": cfg, "numpy": np.__version__}
    for filename in ["cpu.max", "memory.max"]:
        p = Path('/sys/fs/cgroup') / filename
        environment[filename] = p.read_text().strip() if p.exists() else None
    environment["python_packages"] = subprocess.check_output(["python", "-m", "pip", "freeze"], text=True).splitlines()
    write_json(OUT / "environment.json", environment)
    # 先进行一次不计入汇总的预热，降低首次文件缓存/动态库加载影响。
    with tempfile.TemporaryDirectory(prefix="ngram-warm-") as temp:
        warm = ["lmplz", "-o", "3", "-S", "512M", "-T", temp,
                "--text", str(DATA / "train.txt"), "--arpa", str(Path(temp) / "warm.arpa"), "--discount_fallback"]
        run_timed(warm, "warmup", cfg["timeout_seconds"])
    rng = random.Random(cfg["seed"])
    runs, order_log = [], []
    for repeat in range(1, cfg["repeats"] + 1):
        specs = list(cfg["models"])
        rng.shuffle(specs)
        for spec in specs:
            label = f"{spec['id']}_r{repeat}"
            arpa = MODELS / f"{spec['id']}.arpa"
            binary = MODELS / f"{spec['id']}.bin"
            print(f"训练 {label}: n={spec['order']} 内存预算={spec['memory']} 剪枝={spec['prune']}", flush=True)
            with tempfile.TemporaryDirectory(prefix="ngram-sort-") as temp:
                command = ["lmplz", "-o", str(spec["order"]), "-S", spec["memory"], "-T", temp,
                           "--text", str(DATA / "train.txt"), "--arpa", str(arpa), "--discount_fallback",
                           "--prune", *map(str, spec["prune"])]
                train = run_timed(command, label, cfg["timeout_seconds"])
            # KenLM 拒绝覆盖已有 binary；仅替换这个实验自己的输出文件。
            binary.unlink(missing_ok=True)
            build = run_timed(["build_binary", "probing", str(arpa), str(binary)], label + "_binary", cfg["timeout_seconds"])
            record = {**spec, "repeat": repeat, "train": train, "build": build,
                      "arpa_sha256": sha256(arpa), "arpa_mib": arpa.stat().st_size / 2**20,
                      "binary_mib": binary.stat().st_size / 2**20, "ngram_counts": arpa_counts(arpa)}
            text = (ROOT / train["log"]).read_text(encoding="utf-8")
            record["discount_fallback_used"] = "substitut" in text.lower() or "falling back" in text.lower()
            write_json(OUT / f"runs/{label}.json", record)
            runs.append(record)
            order_log.append(label)
    summaries = []
    for spec in cfg["models"]:
        subset = [r for r in runs if r["id"] == spec["id"]]
        times = [r["train"]["elapsed_seconds"] for r in subset]
        model = kenlm.Model(str(MODELS / f"{spec['id']}.bin"))
        summary = {**spec, "train_seconds_median": statistics.median(times),
                   "train_seconds_min": min(times), "train_seconds_max": max(times),
                   "train_seconds_all": times, "peak_rss_mib_max": max(r["train"]["peak_rss_kib"] for r in subset) / 1024,
                   "build_seconds_median": statistics.median(r["build"]["elapsed_seconds"] for r in subset),
                   "arpa_mib": subset[-1]["arpa_mib"], "binary_mib": subset[-1]["binary_mib"],
                   "ngram_counts": subset[-1]["ngram_counts"],
                   "deterministic_arpa": len({r["arpa_sha256"] for r in subset}) == 1,
                   "discount_fallback_used": any(r["discount_fallback_used"] for r in subset),
                   "dev": evaluate(model, DATA / "dev.txt")}
        summaries.append(summary)
    # 只根据验证集选型，冻结后才读取测试集分数；内存变体不重复参加选型。
    eligible = [s for s in summaries if "_mem" not in s["id"]]
    winner = min(eligible, key=lambda s: (s["dev"]["ppl"], s["binary_mib"]))
    write_json(OUT / "selection.json", {"selected": winner["id"], "criterion": "最低验证集 PPL；平手选更小模型",
                                        "dev_ppl": winner["dev"]["ppl"], "frozen_before_test": True})
    for summary in summaries:
        model = kenlm.Model(str(MODELS / f"{summary['id']}.bin"))
        summary["test"] = evaluate(model, DATA / "test.txt")
    write_json(OUT / "summary.json", summaries)
    write_json(OUT / "run_order.json", order_log)
    write_json(OUT / "timing.json", {"experiment_seconds": time.perf_counter() - started,
                                    "training_calls": len(runs), "warmup_calls": 1})
    with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["id", "order", "memory", "prune", "train_seconds_median", "train_seconds_min", "train_seconds_max",
                  "peak_rss_mib_max", "build_seconds_median", "binary_mib", "dev_ppl", "test_ppl"]
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s["dev"]["ppl"] if k == "dev_ppl" else s["test"]["ppl"] if k == "test_ppl" else s[k] for k in fields})
    print(f"完成 {len(runs)} 次训练，验证集选择 {winner['id']}", flush=True)
    return summaries


def sample_distribution(log10_scores, temperature=1.0, top_k=0, greedy=False):
    """温度作用于自然对数概率；top-k 后重新归一化。稳定处理很小的概率。"""
    if temperature <= 0 or top_k < 0:
        raise ValueError("temperature 必须 > 0，top_k 必须 >= 0")
    scores = np.asarray(log10_scores, dtype=np.float64)
    if not np.isfinite(scores).all() or not len(scores):
        raise ValueError("候选分数必须非空且有限")
    if greedy:
        p = np.zeros(len(scores))
        p[np.argmax(scores)] = 1.0
        return p
    logits = scores * math.log(10) / temperature
    if 0 < top_k < len(logits):
        # 稳定排序让相同分数的候选按固定词表顺序打破平手。
        indices = np.argsort(-logits, kind="stable")[:top_k]
        masked = np.full(len(logits), -np.inf)
        masked[indices] = logits[indices]
        logits = masked
    p = np.exp(logits - np.max(logits))
    return p / p.sum()


def generate(model_id, prompt=None, temperature=1.0, top_k=20, seed=11, max_tokens=60, greedy=False):
    import kenlm
    if max_tokens < 1:
        raise ValueError("max_tokens 必须 >= 1")
    model = kenlm.Model(str(MODELS / f"{model_id}.bin"))
    vocabulary = read_json(DATA / "vocabulary.json")
    vocab_set = set(vocabulary)
    prompt = prompt or read_json(ROOT / "config.json")["prompt_tokens"]
    mapped = [w if w in vocab_set else RARE for w in prompt]
    words = sorted(w for w in vocabulary if w != RARE) + ["</s>"]
    state = kenlm.State()
    model.BeginSentenceWrite(state)
    for token in mapped:
        nxt = kenlm.State()
        model.BaseScore(state, token, nxt)
        state = nxt
    rng = np.random.default_rng(seed)
    produced, trace = [], []
    scratch = kenlm.State()
    started = time.perf_counter()
    for step in range(max_tokens):
        scores = np.fromiter((model.BaseScore(state, w, scratch) for w in words), dtype=np.float64, count=len(words))
        p = sample_distribution(scores, temperature, top_k, greedy)
        index = int(np.argmax(p)) if greedy else int(rng.choice(len(words), p=p))
        word = words[index]
        top = np.argsort(-p, kind="stable")[:5]
        trace.append({"step": step + 1, "token": word, "model_log10_probability": float(scores[index]),
                      "sampling_probability": float(p[index]),
                      "top5": [{"token": words[i], "probability": float(p[i])} for i in top]})
        if word == "</s>":
            break
        produced.append(word)
        nxt = kenlm.State()
        model.BaseScore(state, word, nxt)
        state = nxt
    bigrams = list(zip(produced, produced[1:]))
    return {"model": model_id, "prompt": prompt, "mapped_prompt": mapped,
            "prompt_rare_words": [w for w in prompt if w not in vocab_set],
            "temperature": temperature, "top_k": top_k, "seed": seed, "greedy": greedy,
            "generated_tokens": produced, "continuation": "".join(produced),
            "text": "".join(prompt + produced), "length": len(produced),
            "ended_with_eos": trace[-1]["token"] == "</s>",
            "distinct2": len(set(bigrams)) / len(bigrams) if bigrams else None,
            "seconds": time.perf_counter() - started, "trace": trace}


def run_generations():
    cfg = settings()
    winner = read_json(OUT / "selection.json")["selected"]
    specs = [dict(label="低温_k20", temperature=0.7, top_k=20),
             dict(label="基准_k20", temperature=1.0, top_k=20),
             dict(label="高温_k20", temperature=1.3, top_k=20),
             dict(label="基准_k5", temperature=1.0, top_k=5),
             dict(label="全词表", temperature=1.0, top_k=0)]
    samples = []
    for spec in specs:
        for seed in cfg["generation_seeds"]:
            print(f"续写 {winner} {spec['label']} seed={seed}", flush=True)
            samples.append({"setting": spec["label"], **generate(winner, temperature=spec["temperature"], top_k=spec["top_k"],
                           seed=seed, max_tokens=cfg["generation_max_tokens"])})
    samples.append({"setting": "贪心", **generate(winner, greedy=True, max_tokens=cfg["generation_max_tokens"])})
    # 相同解码参数下观察阶数变化，不依据样例美观程度重新选型。
    for model_id in ["kn2", "kn3", "kn4", "kn5"]:
        for seed in [11, 22]:
            samples.append({"setting": "阶数对比", **generate(model_id, seed=seed, max_tokens=cfg["generation_max_tokens"])})
    write_json(OUT / "generations.json", samples)
    lines = ["# KenLM 固定种子续写（全部保留）", "", "开头的省略号作为待续写位置，不输入模型；原始开头按词级切分。",
             "所有结果均为程序原样输出，不手工挑选、润色或强行禁止 EOS。", ""]
    for i, s in enumerate(samples, 1):
        lines += [f"## {i}. {s['model']} / {s['setting']} / seed={s['seed']}", "",
                  f"温度={s['temperature']}；top-k={s['top_k']}；生成 {s['length']} 词；EOS={s['ended_with_eos']}。", "",
                  "> " + s["text"], ""]
    (OUT / "生成样例.md").write_text("\n".join(lines), encoding="utf-8")
    return samples
