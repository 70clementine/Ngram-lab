"""全词表概率向量 -> 禁用特殊词 -> 温度 -> top-k -> 采样。

向量化只是加速，单词概率与 NLTK score 一致，tests 中有独立核对。
"""
from collections import OrderedDict
import time
import numpy as np

from .common import ROOT, config, read_json, write_json
from .modeling import BOS, EOS, UNK, load_model


class Distribution:
    def __init__(self, model, spec):
        self.model, self.spec = model, spec
        self.words = sorted(set(model.vocab))
        self.index = {w: i for i, w in enumerate(self.words)}
        self.cache = OrderedDict()

    def probabilities(self, context):
        context = tuple(self.model.vocab.lookup(context))
        if context in self.cache:
            self.cache.move_to_end(context)
            return self.cache[context].copy()
        counts = self.model.context_counts(context)
        vector = np.zeros(len(self.words), dtype=np.float64)
        for word, count in counts.items():
            vector[self.index[word]] = count
        total = counts.N()
        kind = self.spec["smoothing"]
        if kind == "lidstone":
            gamma = self.spec["gamma"]
            vector = (vector + gamma) / (total + gamma * len(self.model.vocab))
        elif kind == "mle" or not context:
            if total:
                vector /= total
        elif not total:
            vector = self.probabilities(context[1:])
        else:
            types = sum(c > 0 for c in counts.values())
            vector = vector / (total + types) + types / (total + types) * self.probabilities(context[1:])
        # 只缓存少量上下文，避免生成时占用数 GB 内存。
        if len(self.cache) >= 64:
            self.cache.popitem(last=False)
        self.cache[context] = vector.copy()
        return vector

    def sample(self, context, rng, temperature=1.0, top_k=0, greedy=False):
        if temperature <= 0 or top_k < 0:
            raise ValueError("temperature 必须大于零，top_k 必须非负")
        p = self.probabilities(context)
        for token in [BOS, UNK]:
            p[self.index[token]] = 0
        if not p.sum():
            raise ValueError("当前上下文无可生成词；请使用平滑模型")
        if greedy:
            idx = int(np.argmax(p))
        else:
            valid = np.flatnonzero(p > 0)
            logp = np.log(p[valid]) / temperature
            weights = np.exp(logp - logp.max())
            if top_k and top_k < len(valid):
                # 稳定排序保证并列概率时同一平台也能确定复现。
                keep = np.argsort(-weights, kind="stable")[:top_k]
                valid, weights = valid[keep], weights[keep]
            idx = int(rng.choice(valid, p=weights / weights.sum()))
        return self.words[idx]


def generate(model, spec, prompt, seed=11, temperature=1.0, top_k=0, greedy=False, max_tokens=60, distribution=None):
    start = time.perf_counter()
    dist = distribution or Distribution(model, spec)
    history = [BOS] * (model.order - 1) + list(model.vocab.lookup(prompt))
    words, ended = [], False
    rng = np.random.default_rng(seed)
    for _ in range(max_tokens):
        context = tuple(history[-(model.order - 1):]) if model.order > 1 else ()
        word = dist.sample(context, rng, temperature, top_k, greedy)
        if word == EOS:
            ended = True
            break
        words.append(word)
        history.append(word)
    bigrams = list(zip(words, words[1:]))
    distinct2 = len(set(bigrams)) / len(bigrams) if bigrams else None
    return {"model": spec["id"], "seed": seed, "temperature": temperature, "top_k": top_k,
            "greedy": greedy, "prompt_tokens": prompt,
            "prompt_oov": [w for w in prompt if w not in model.vocab],
            "generated_tokens": words, "text": "".join(prompt + words),
            "length": len(words), "ended_with_eos": ended, "distinct2": distinct2,
            "repeated_bigram_fraction": 1 - distinct2 if distinct2 is not None else None,
            "seconds": time.perf_counter() - start}


def run_generation():
    cfg = config()
    selection = read_json(ROOT / "results/selection.json")
    chosen = selection["best_full_data_wb_id"]
    summaries = {c["id"]: c for c in read_json(ROOT / "results/summary.json")}
    # 解码实验只改变一个因素；额外的阶数对照使用相同 T/k/seed。
    settings = [("greedy", 1.0, 0, True), ("T0.7_k20", 0.7, 20, False),
                ("T1.0_k20", 1.0, 20, False), ("T1.3_k20", 1.3, 20, False),
                ("T1.0_k5", 1.0, 5, False), ("T1.0_full", 1.0, 0, False)]
    results = []
    for model_id in sorted({"wb_n1", "wb_n2", "wb_n3", "wb_n4", chosen}):
        model = load_model(model_id)
        spec = summaries[model_id]
        dist = Distribution(model, spec)
        use_settings = settings if model_id == chosen else [("T1.0_k20", 1.0, 20, False)]
        for label, temp, top_k, greedy in use_settings:
            seeds = cfg["generation_seeds"][:1] if greedy else cfg["generation_seeds"]
            for seed in seeds:
                row = generate(model, spec, cfg["prompt_tokens"], seed, temp, top_k, greedy,
                               cfg["generation_max_tokens"], dist)
                row["setting"] = label
                results.append(row)
        print(f"generated: {model_id}", flush=True)
    write_json(ROOT / "results/generations.json", results)
    model = load_model(chosen)
    dist = Distribution(model, summaries[chosen])
    mapped = list(model.vocab.lookup(cfg["prompt_tokens"]))
    context = tuple(mapped[-(model.order - 1):]) if model.order > 1 else ()
    stages = []
    for drop in range(len(context) + 1):
        suffix = context[drop:]
        counts = model.context_counts(suffix)
        stages.append({"context": list(suffix), "total": counts.N(), "types": len(counts),
                       "observed_top5": counts.most_common(5)})
    p = dist.probabilities(context)
    ranked = sorted(zip(dist.words, p.tolist()), key=lambda pair: -pair[1])[:10]
    write_json(ROOT / "results/prediction_trace.json", {"model": chosen, "context": list(context),
               "stages": stages, "raw_probability_top10": [{"word": w, "probability": p} for w,p in ranked],
               "note": "原始模型概率，尚未屏蔽特殊词、调温或进行 top-k 截断"})
    return results
