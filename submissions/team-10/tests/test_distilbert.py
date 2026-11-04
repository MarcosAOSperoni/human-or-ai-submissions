"""DistilBERT classifier mechanics with a tiny randomly initialized model (fast, CPU).

Needs the distilbert-base-uncased tokenizer in the local Hugging Face cache
(downloaded by the first training run); skipped otherwise.
"""

import copy

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")
from transformers import AutoTokenizer, DistilBertConfig as HFConfig, DistilBertForSequenceClassification  # noqa: E402

from src.classifiers import TextClassifier  # noqa: E402
from src.classifiers.distilbert import DistilBertClassifier, predict_proba_from_checkpoint  # noqa: E402

try:
    TOKENIZER = AutoTokenizer.from_pretrained("distilbert-base-uncased", local_files_only=True)
except OSError:
    TOKENIZER = None
pytestmark = pytest.mark.skipif(TOKENIZER is None, reason="distilbert tokenizer not cached")

TEXTS = ["the cat sat on the mat", "furthermore, it is important to note", "i dunno lol",
         "in conclusion, this essay has shown", "we went to the beach yesterday", "moreover, a delve into"] * 4
LABELS = [0, 1, 0, 1, 0, 1] * 4


def tiny_classifier(**overrides):
    clf = DistilBertClassifier(device="cpu", epochs=2, batch_size=8, max_length=32, **overrides)
    clf.tokenizer = copy.deepcopy(TOKENIZER)  # fit() may add a newline-marker token
    cfg = HFConfig(vocab_size=TOKENIZER.vocab_size, dim=32, hidden_dim=64, n_layers=1, n_heads=2, num_labels=2)
    clf.model = DistilBertForSequenceClassification(cfg)
    return clf


def test_fit_predict_shapes_and_history(tmp_path):
    clf = tiny_classifier()
    assert isinstance(clf, TextClassifier)
    clf.fit(TEXTS, LABELS, TEXTS[:12], LABELS[:12], checkpoint_dir=tmp_path / "ckpt", log=lambda s: None)
    assert [h["epoch"] for h in clf.history] == [1, 2]
    assert all({"train_loss", "val_loss", "val_auc"} <= h.keys() for h in clf.history)
    assert clf.best_epoch in (1, 2)
    p = clf.predict_proba(TEXTS)
    assert p.shape == (len(TEXTS),) and np.all((p >= 0) & (p <= 1))


def test_order_preserved_and_checkpoint_roundtrip(tmp_path):
    clf = tiny_classifier()
    clf.fit(TEXTS, LABELS, TEXTS[:6], LABELS[:6], checkpoint_dir=tmp_path / "ckpt", log=lambda s: None)
    p = clf.predict_proba(TEXTS)
    np.testing.assert_allclose(clf.predict_proba(TEXTS[::-1])[::-1], p, atol=1e-6)
    np.testing.assert_allclose(predict_proba_from_checkpoint(TEXTS, tmp_path / "ckpt", device="cpu"), p, atol=1e-6)
    loaded = DistilBertClassifier.load(tmp_path / "ckpt", device="cpu")
    assert loaded.config.max_length == 32 and loaded.best_epoch == clf.best_epoch
    assert len(loaded.history) == loaded.best_epoch  # metadata saved when the best epoch was reached


def test_seed_reproducible():
    a, b = tiny_classifier(), tiny_classifier()
    b.model.load_state_dict(a.model.state_dict())
    a.fit(TEXTS, LABELS, log=lambda s: None)
    b.fit(TEXTS, LABELS, log=lambda s: None)
    np.testing.assert_allclose(a.predict_proba(TEXTS), b.predict_proba(TEXTS), atol=1e-6)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        DistilBertClassifier(device="cpu").predict_proba(["x"])


def test_newline_marker_is_one_token_and_survives_checkpoint(tmp_path):
    texts = [t.replace(" ", "\n", 1) for t in TEXTS]
    clf = tiny_classifier(newline_marker="[NL]")
    clf.fit(texts, LABELS, texts[:6], LABELS[:6], checkpoint_dir=tmp_path / "ckpt", log=lambda s: None)
    nl_id = clf.tokenizer.convert_tokens_to_ids("[NL]")
    assert nl_id == TOKENIZER.vocab_size  # appended after the original vocabulary
    assert clf.model.get_input_embeddings().num_embeddings == len(clf.tokenizer)
    assert clf._encode(["a\nb"])["input_ids"][0].count(nl_id) == 1
    assert clf._encode(["a b"])["input_ids"][0].count(nl_id) == 0
    loaded = DistilBertClassifier.load(tmp_path / "ckpt", device="cpu")
    assert loaded.config.newline_marker == "[NL]"
    np.testing.assert_allclose(loaded.predict_proba(texts), clf.predict_proba(texts), atol=1e-6)
