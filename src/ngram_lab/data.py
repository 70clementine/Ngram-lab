"""下载、严格解析 POS 标注、按文章抽样划分；不使用测试集建立词表。"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import random
import re
import time
import urllib.request

from .common import ROOT, config, write_json, sha256

SOURCE_URL = "https://ndownloader.figshare.com/files/10193073"
SOURCE_SHA256 = "1e2574641b92bc07c61af95162ce3c62bdef5a5da04ceff0d50144b4aae4b6a1"
ID = re.compile(r"^(\d{8}-\d{2}-\d{3})-\d{3}/m$")
TAG = re.compile(r"(?P<word>.+)/(?P<pos>[A-Za-z]+)(?:\][A-Za-z]+)*$")
END = {"。", "！", "？", "!", "?"}
CLOSERS = {"”", "’", "）", "》", "」", "』", '"', ")"}
# 仅统一全角数字/英文字母，不把中文句号、逗号删除或变成空白。
WIDTH = str.maketrans({chr(x): chr(x - 0xFEE0) for a, b in [(0xFF10, 0xFF19), (0xFF21, 0xFF3A), (0xFF41, 0xFF5A)] for x in range(a, b + 1)})


def download():
    path = ROOT / "data/raw/people_daily_1998.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # 写临时文件，下载完整且校验通过后才成为输入语料。
        temp = path.with_suffix(".download")
        urllib.request.urlretrieve(SOURCE_URL, temp)
        if sha256(temp) != SOURCE_SHA256:
            raise ValueError("下载校验失败，拒绝使用不同版本的数据")
        temp.replace(path)
    if sha256(path) != SOURCE_SHA256:
        raise ValueError("本地语料 SHA256 不匹配；请检查文件是否被修改")
    metadata = ROOT / "data/raw/source_metadata.json"
    if not metadata.exists():
        with urllib.request.urlopen("https://api.figshare.com/v2/articles/5777397", timeout=60) as response:
            metadata.write_bytes(response.read())
    return path


def parse_line(line):
    """仅支持本次已校验的 PFR 格式；有不认识的条目就报错，不静默删除。"""
    parts = line.split()
    match = ID.fullmatch(parts[0]) if parts else None
    if not match:
        raise ValueError(f"无效文章/段落编号: {line[:80]}")
    words = []
    for token in parts[1:]:
        match_token = TAG.fullmatch(token)
        if not match_token:
            raise ValueError(f"无法解析标注条目: {token}")
        word = match_token["word"]
        # [/w 是文字中的方括号；[中央/n 中的 [ 才是实体包装。
        if word.startswith("[") and word != "[":
            word = word.lstrip("[")
        word = word.translate(WIDTH)
        if not word or word in {"<s>", "</s>", "<UNK>"}:
            raise ValueError(f"非法正文 token: {token}")
        words.append(word)
    return match[1], words


def split_sentences(words):
    """句末引号归入前句；段落末尾即使没有句号也结束一个片段。"""
    current, ended = [], False
    for word in words:
        if ended and word not in CLOSERS:
            yield current
            current = []
            ended = False
        current.append(word)
        ended = ended or word in END
    if current:
        yield current


def prepare():
    started = time.perf_counter()
    cfg = config()
    raw = download()
    text = raw.read_bytes().decode("gb18030", errors="strict")
    documents = defaultdict(list)
    raw_lines, raw_words, entity_openings = 0, 0, 0
    examples = []
    for line in text.splitlines():
        if not line.strip():
            continue
        doc_id, words = parse_line(line)
        raw_lines += 1
        raw_words += len(words)
        entity_openings += sum(t.startswith("[") and not t.startswith("[/w") for t in line.split()[1:])
        documents[doc_id].extend(split_sentences(words))
        if len(examples) < 4 and ("[" in line or len(examples) == 0):
            # 完整对照保存在本地 data 中；公开报告只用自造教学例句。
            examples.append({"raw": line, "clean": " ".join(words)})

    # 在划分前做全局精确句子去重；按文章 ID 排序，确保去重稳定可复现。
    seen, clean, duplicates = set(), {}, 0
    for doc_id in sorted(documents):
        kept = []
        for sent in documents[doc_id]:
            key = tuple(sent)
            if key in seen:
                duplicates += 1
            else:
                seen.add(key)
                kept.append(sent)
        if kept:
            clean[doc_id] = kept
    doc_ids = sorted(clean)
    rng = random.Random(cfg["seed"])
    rng.shuffle(doc_ids)
    selected, tokens = [], 0
    for doc_id in doc_ids:
        size = sum(map(len, clean[doc_id]))
        if tokens + size <= cfg["max_corpus_tokens"]:
            selected.append(doc_id)
            tokens += size
    if len(selected) < 10:
        raise ValueError("抽样不足 10 篇文章，请提高 max_corpus_tokens")
    # 抽样后重新洗牌，防止只给验证/测试集留下凑预算的短文。
    rng.shuffle(selected)
    ntrain = int(len(selected) * cfg["train_fraction"])
    ndev = int(len(selected) * cfg["dev_fraction"])
    groups = {"train": selected[:ntrain], "dev": selected[ntrain:ntrain + ndev], "test": selected[ntrain + ndev:]}
    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    corpus = {name: [s for d in ids for s in clean[d]] for name, ids in groups.items()}
    counts = Counter(w for s in corpus["train"] for w in s)
    vocab = sorted(w for w, n in counts.items() if n >= cfg["min_word_count"])
    vocab_set = set(vocab)
    stats = {}
    manifests = {}
    for name, sents in corpus.items():
        nwords = sum(map(len, sents))
        mapped = [[w if w in vocab_set else "<UNK>" for w in s] for s in sents]
        (out / f"{name}.txt").write_text("\n".join(" ".join(s) for s in mapped) + "\n", encoding="utf-8")
        (out / f"{name}_original.txt").write_text("\n".join(" ".join(s) for s in sents) + "\n", encoding="utf-8")
        stats[name] = {"articles": len(groups[name]), "sentences": len(sents), "tokens": nwords,
                       "unk_tokens": sum(w not in vocab_set for s in sents for w in s),
                       "raw_oov_tokens": sum(w not in counts for s in sents for w in s),
                       "file_sha256": sha256(out / f"{name}.txt")}
        # 只含 ID、句数、词数，不公开语料正文。
        manifests[name] = [{"id": d, "sentences": len(clean[d]), "tokens": sum(map(len, clean[d]))} for d in groups[name]]
    write_json(out / "vocabulary.json", vocab)
    write_json(out / "cleaning_examples.json", examples)
    write_json(ROOT / "results/split_manifest.json", manifests)
    summary = {"source_url": SOURCE_URL, "source_doi": "10.6084/m9.figshare.5777397.v1",
               "source_sha256": SOURCE_SHA256, "source_bytes": raw.stat().st_size,
               "source_encoding": "gb18030 (strict)", "source_nonempty_lines": raw_lines,
               "source_tokens_without_ids": raw_words, "source_articles": len(documents),
               "source_date_min": min(documents)[:8], "source_date_max": max(documents)[:8],
               "removed_entity_openings": entity_openings, "duplicate_sentences_removed": duplicates,
               "unique_sentences_before_sampling": len(seen), "sampled_tokens": tokens,
               "sampled_articles": len(selected), "lexical_vocabulary_size": len(vocab),
               "min_word_count": cfg["min_word_count"], "splits": stats,
               "elapsed_seconds": time.perf_counter() - started,
               "created_utc": datetime.now(timezone.utc).isoformat(), "config": cfg}
    write_json(ROOT / "results/data_summary.json", summary)
    print(f"prepared: {tokens:,} tokens, {len(selected)} articles; vocabulary={len(vocab):,}", flush=True)
    return summary
