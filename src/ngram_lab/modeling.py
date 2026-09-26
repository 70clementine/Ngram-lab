"""NLTK 模型、训练计时、逐预测事件评测。没有 epoch 或梯度下降。"""
from collections import Counter
from functools import lru_cache
import math
import os
import pickle
import threading
import time

import psutil
from nltk.lm import MLE, Lidstone, WittenBellInterpolated, Vocabulary

from .common import ROOT, read_json, write_json, sentences

BOS, EOS, UNK = "<s>", "</s>", "<UNK>"


def configurations():
    configs = [{"id": f"wb_n{n}", "order": n, "smoothing": "witten_bell", "gamma": None, "fraction": 1.0} for n in [1, 2, 3, 4]]
    configs += [{"id": "mle_n3", "order": 3, "smoothing": "mle", "gamma": None, "fraction": 1.0}]
    for gamma in [0.01, 0.1, 1.0]:
        configs.append({"id": f"lid_n3_g{gamma:g}", "order": 3, "smoothing": "lidstone", "gamma": gamma, "fraction": 1.0})
    for fraction in [0.25, 0.5]:
        configs.append({"id": f"lid_n3_g0.1_f{fraction:g}", "order": 3, "smoothing": "lidstone", "gamma": 0.1, "fraction": fraction})
    return configs


def make_model(spec, vocabulary):
    # 不把 UNK 显式放进 Vocabulary.counts：NLTK 会自动附加一次 UNK。
    vocab = Vocabulary([w for w in vocabulary if w != UNK] + [BOS, EOS], unk_cutoff=1, unk_label=UNK)
    if spec["smoothing"] == "mle":
        return MLE(spec["order"], vocabulary=vocab)
    if spec["smoothing"] == "lidstone":
        return Lidstone(spec["gamma"], spec["order"], vocabulary=vocab)
    return WittenBellInterpolated(spec["order"], vocabulary=vocab)


def sentence_ngrams(words, order):
    """每个正文词和一次 EOS 各贡献 1..n 阶计数；BOS 只作上下文。

    不直接对两端重复填充后的全部 everygrams 计分，避免重复 EOS 事件。
    例：2 阶 ['我','来'] -> ('我',),('<s>','我'),('来',),('我','来'),
    ('</s>',),('来','</s>')。
    """
    padded = [BOS] * (order - 1) + list(words) + [EOS]
    for i in range(order - 1, len(padded)):
        for n in range(1, order + 1):
            yield tuple(padded[i - n + 1:i + 1])


def training_sentences(fraction):
    all_sents = sentences("train")
    articles = read_json(ROOT / "results/split_manifest.json")["train"]
    count = max(1, int(len(articles) * fraction))
    # 使用同一随机文章顺序的前缀，各规模严格嵌套，而且不截断文章。
    nsents = sum(a["sentences"] for a in articles[:count])
    return all_sents[:nsents], count


def evaluate(model, sents):
    """整套数据按词加权聚合 NLL；不是先算各句 PPL 再取平均。"""
    score = lru_cache(maxsize=100_000)(model.score)
    nll, events, zeros, unknowns = 0.0, 0, 0, 0
    start = time.perf_counter()
    for words in sents:
        history = [BOS] * (model.order - 1)
        for word in words + [EOS]:
            context = tuple(history[-(model.order - 1):]) if model.order > 1 else ()
            p = score(word, context)
            if not math.isfinite(p) or p < 0 or p > 1 + 1e-10:
                raise ValueError(f"非法概率 {p}")
            if p == 0:
                zeros += 1
            else:
                nll -= math.log(p)
            unknowns += word == UNK
            events += 1
            history.append(word)
    score.cache_clear()
    # JSON 不写非标准 Infinity；以 null + 明确的 infinity 标志表示。
    entropy = None if zeros else nll / events
    return {"events": events, "zero_events": zeros, "unk_events": unknowns,
            "cross_entropy_nats": entropy, "ppl": math.exp(entropy) if entropy is not None else None,
            "ppl_infinite": bool(zeros), "seconds": time.perf_counter() - start}


def train_worker(model_id, repeat):
    spec = next(c for c in configurations() if c["id"] == model_id)
    sents, narticles = training_sentences(spec["fraction"])
    vocabulary = read_json(ROOT / "data/processed/vocabulary.json")
    model = make_model(spec, vocabulary)
    proc = psutil.Process(os.getpid())
    baseline = proc.memory_info().rss
    peak = [baseline]
    done = threading.Event()

    def monitor():
        while not done.wait(0.02):
            peak[0] = max(peak[0], proc.memory_info().rss)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    start = time.perf_counter()
    progress = []
    processed = 0

    def stream():
        nonlocal processed
        for i, words in enumerate(sents, 1):
            yield sentence_ngrams(words, model.order)
            processed += len(words)
            if i % 500 == 0 or i == len(sents):
                row = {"sentences": i, "tokens": processed, "seconds": time.perf_counter() - start,
                       "rss_mb": proc.memory_info().rss / 2**20}
                progress.append(row)
                print(f"{model_id} repeat={repeat}: {i}/{len(sents)} sentences, {row['seconds']:.2f}s, {row['rss_mb']:.1f} MiB", flush=True)

    model.fit(stream())
    elapsed = time.perf_counter() - start
    peak[0] = max(peak[0], proc.memory_info().rss)
    done.set()
    thread.join()
    distinct = {}
    for n in range(1, model.order + 1):
        distinct[str(n)] = len(model.counts.unigrams) if n == 1 else sum(len(freq) for freq in model.counts[n].values())
    result = {**spec, "repeat": repeat, "train_seconds": elapsed, "train_tokens": processed,
              "train_sentences": len(sents), "train_articles": narticles,
              "tokens_per_second": processed / elapsed, "peak_rss_mb": peak[0] / 2**20,
              "baseline_rss_mb": baseline / 2**20, "incremental_rss_mb": (peak[0] - baseline) / 2**20,
              "distinct_ngrams": distinct, "progress": progress}
    if repeat == 1:
        model_path = ROOT / "models" / f"{model_id}.pkl"
        model_path.parent.mkdir(exist_ok=True)
        save_start = time.perf_counter()
        with model_path.open("wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        result["save_seconds"] = time.perf_counter() - save_start
        result["model_mb"] = model_path.stat().st_size / 2**20
        result["dev"] = evaluate(model, sentences("dev"))
    write_json(ROOT / "results/runs" / f"{model_id}_r{repeat}.json", result)


def load_model(model_id):
    # 仅加载本项目自己训练的模型；pickle 不是第三方模型交换格式。
    with (ROOT / "models" / f"{model_id}.pkl").open("rb") as f:
        return pickle.load(f)
