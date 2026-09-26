"""验证 PPL 分母、KenLM 状态分数、采样变换与实际模型的一致性。"""
import math
import numpy as np
import pytest

from ngram_lab.kenlm_pipeline import sample_distribution, evaluate, DATA, MODELS, OUT, RARE
from ngram_lab.common import read_json, sha256


def test_temperature_and_topk_math():
    scores = np.log10([0.6, 0.3, 0.1])
    np.testing.assert_allclose(sample_distribution(scores), [0.6, 0.3, 0.1])
    np.testing.assert_allclose(sample_distribution(scores, top_k=2), [2/3, 1/3, 0])
    np.testing.assert_allclose(sample_distribution(scores, temperature=0.5), np.array([0.36, 0.09, 0.01])/0.46)
    np.testing.assert_array_equal(sample_distribution(scores, greedy=True), [1, 0, 0])
    with pytest.raises(ValueError):
        sample_distribution(scores, temperature=0)
    with pytest.raises(ValueError):
        sample_distribution(scores, top_k=-1)


def test_inputs_and_fixed_vocabulary():
    if not (DATA / "train.txt").exists():
        pytest.skip("先运行 KenLM prepare")
    metadata = read_json(OUT / "data.json")
    vocabulary = set(read_json(DATA / "vocabulary.json"))
    for split in ["train", "dev", "test"]:
        path = DATA / f"{split}.txt"
        assert sha256(path) == metadata["splits"][split]["sha256"]
        words = path.read_text(encoding="utf-8").split()
        assert set(words) <= vocabulary
        assert not set(words) & {"<s>", "</s>", "<unk>", "<UNK>"}
    assert set((DATA / "train.txt").read_text(encoding="utf-8").split()) == vocabulary


def test_kenlm_state_score_and_ppl_agree(tmp_path):
    kenlm = pytest.importorskip("kenlm", reason="KenLM Python 绑定在 Docker 中")
    if not (MODELS / "kn3.bin").exists():
        pytest.skip("先训练 KenLM")
    model = kenlm.Model(str(MODELS / "kn3.bin"))
    sentence = "我们 学校 召开 了 会议 。"
    state = kenlm.State()
    model.BeginSentenceWrite(state)
    total = 0.0
    for word in sentence.split() + ["</s>"]:
        nxt = kenlm.State()
        total += model.BaseScore(state, word, nxt)
        state = nxt
    assert total == pytest.approx(model.score(sentence, bos=True, eos=True), abs=1e-5)
    path = tmp_path / "one.txt"
    path.write_text(sentence + "\n", encoding="utf-8")
    result = evaluate(model, path)
    assert result["events"] == len(sentence.split()) + 1
    assert result["ppl"] == pytest.approx(model.perplexity(sentence), rel=1e-5)
    # 独立验证每个条件分布在完整词表和 EOS 上归一化（<s> 不可预测）。
    vocabulary = read_json(DATA / "vocabulary.json") + ["</s>", "<unk>"]
    scratch = kenlm.State()
    for history in [[], ["召开", "了"], ["从未见过的词XYZ"]]:
        model.BeginSentenceWrite(state)
        for word in history:
            nxt = kenlm.State()
            model.BaseScore(state, word, nxt)
            state = nxt
        mass = sum(10 ** model.BaseScore(state, w, scratch) for w in vocabulary)
        assert mass == pytest.approx(1.0, abs=2e-4)


def test_kenlm_generation_reproducibility():
    pytest.importorskip("kenlm")
    if not (MODELS / "kn3.bin").exists():
        pytest.skip("先训练 KenLM")
    from ngram_lab.kenlm_pipeline import generate
    a = generate("kn3", seed=22, max_tokens=8)
    b = generate("kn3", seed=22, max_tokens=8)
    assert a["generated_tokens"] == b["generated_tokens"]
    assert not set(a["generated_tokens"]) & {RARE, "<s>", "</s>", "<unk>"}
