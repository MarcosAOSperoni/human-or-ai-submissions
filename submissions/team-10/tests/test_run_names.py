"""Run names and checkpoint folders for each model family (no model download needed)."""

import pytest

pytest.importorskip("transformers")
from experiments.distilbert_train import model_family, run_name_for  # noqa: E402
from src.classifiers.distilbert import DistilBertConfig  # noqa: E402


@pytest.mark.parametrize("model_name, family", [
    ("distilbert-base-uncased", "distilbert"),
    ("checkpoints/mlm/dapt-split-nl", "distilbert"),
    ("microsoft/deberta-v3-base", "deberta-v3-base"),
    ("models/deberta-v3-large", "deberta-v3-large"),
    ("FacebookAI/roberta-large", "roberta-large"),
    ("models/roberta-large/", "roberta-large"),
])
def test_model_family(model_name, family):
    assert model_family(DistilBertConfig(model_name=model_name)) == family


def test_run_name_tags():
    cfg = DistilBertConfig(model_name="FacebookAI/roberta-large", learning_rate=1e-5, epochs=3,
                           newline_marker="[NL]", warmup_ratio=0.1)
    assert run_name_for(cfg) == "roberta-large_nl_wu0.1_lr1e-05_ep3_bs16_len512_seed42"
    assert run_name_for(DistilBertConfig()) == "distilbert_lr2e-05_ep3_bs16_len512_seed42"
