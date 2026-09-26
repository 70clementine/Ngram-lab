"""检验容易改变实验结论的逻辑：清洗、边界、概率、采样和数据泄漏。"""
import math
import numpy as np
import pytest

from ngram_lab.data import parse_line, split_sentences
from ngram_lab.modeling import make_model, sentence_ngrams, evaluate, BOS, EOS
from ngram_lab.generation import Distribution, generate
from ngram_lab.common import ROOT, read_json, sentences


def test_clean_nested_tags_without_erasing_text_slash():
    _, words = parse_line("19980101-01-001-001/m [北京/ns 大学/n]nt 在/p １９９８年/t 以/p A/B/n 为/v 例/n 。/w")
    assert words == ["北京", "大学", "在", "1998年", "以", "A/B", "为", "例", "。"]
    assert parse_line("19980101-01-001-001/m [/w 甲/n ]/w")[1] == ["[", "甲", "]"]
    with pytest.raises(ValueError):
        parse_line("19980101-01-001-001/m 未知格式")


def test_sentence_closing_quote():
    assert list(split_sentences(["说", "：", "“", "好", "！", "”", "然后", "走", "。"])) == [["说", "：", "“", "好", "！", "”"], ["然后", "走", "。"]]


def toy(kind="mle", order=2, gamma=0.1):
    spec = {"id": "toy", "smoothing": kind, "order": order, "gamma": gamma}
    model = make_model(spec, ["我", "爱", "你", "他"])
    model.fit(sentence_ngrams(s, order) for s in [["我", "爱", "你"], ["我", "爱", "他"], ["<UNK>", "爱", "你"]])
    return model, spec


def test_hand_count_and_single_eos():
    model, _ = toy()
    assert model.counts[["我"]]["爱"] == 2
    assert model.score("你", ["爱"]) == pytest.approx(2 / 3)
    assert model.counts.unigrams[BOS] == 0
    assert model.counts.unigrams[EOS] == 3
    result = evaluate(model, [["我", "爱", "你"]])
    assert result["events"] == 4
    assert result["ppl"] == pytest.approx(math.exp(-math.log((2/3) * 1 * (2/3) * 1) / 4))
    assert evaluate(model, [["你", "我"]])["ppl_infinite"]


@pytest.mark.parametrize("kind", ["mle", "lidstone", "witten_bell"])
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_vector_probabilities_match_independent_nltk_scores(kind, order):
    model, spec = toy(kind, order)
    dist = Distribution(model, spec)
    contexts = [(), ("爱",), ("他", "我"), tuple([BOS] * (order-1))]
    for context in contexts:
        context = context[-(order-1):] if order > 1 else ()
        vector = dist.probabilities(context)
        expected = [model.score(w, context) for w in dist.words]
        np.testing.assert_allclose(vector, expected, rtol=1e-12, atol=1e-12)
        assert vector.sum() == pytest.approx(0 if kind == "mle" and not any(expected) else 1)


def test_seed_and_special_tokens():
    model, spec = toy("witten_bell", 3)
    a = generate(model, spec, ["我"], seed=12)
    b = generate(model, spec, ["我"], seed=12)
    assert a["generated_tokens"] == b["generated_tokens"]
    assert not set(a["generated_tokens"]) & {BOS, EOS, "<UNK>"}


def test_prepared_data_has_no_article_or_sentence_leakage():
    if not (ROOT / "results/split_manifest.json").exists():
        pytest.skip("运行 prepare 后执行数据完整性检查")
    manifest = read_json(ROOT / "results/split_manifest.json")
    names = ["train", "dev", "test"]
    ids = {n: {a["id"] for a in manifest[n]} for n in names}
    # 检查原词序列；UNK 映射后不同句子偶然相同并不等于语料泄漏。
    sets = {n: {tuple(s) for s in sentences(n + "_original")} for n in names}
    for i, a in enumerate(names):
        for b in names[i+1:]:
            assert ids[a].isdisjoint(ids[b])
            assert sets[a].isdisjoint(sets[b])
    for n in names:
        assert not any("/n]nt" in w or w.endswith("/w") for s in sentences(n) for w in s)
