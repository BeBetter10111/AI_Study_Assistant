"""Logic thuần của app/rag/evaluation.py — không cần ragas/LLM."""
import json

import pytest

from app.rag import evaluation as ev


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_load_dataset_jsonl_aliases_and_blank_lines(tmp_path):
    p = _write(tmp_path, "d.jsonl",
               '{"question": "Q1", "ground_truth": "A1", "document_id": "d"}\n\n{"user_input": "Q2"}\n')
    items = ev.load_dataset(p)
    assert [i["question"] for i in items] == ["Q1", "Q2"]
    assert items[0]["reference"] == "A1" and items[1]["reference"] is None
    assert items[1]["response"] is None and items[1]["retrieved_contexts"] is None


def test_load_dataset_json_list_with_prefilled_outputs(tmp_path):
    p = _write(tmp_path, "d.json", json.dumps(
        [{"question": "Q", "response": "R", "retrieved_contexts": ["c1", 2]}]))
    assert ev.load_dataset(p)[0]["retrieved_contexts"] == ["c1", "2"]


@pytest.mark.parametrize("text", ["", '{"answer": "x"}', "{not json"])
def test_load_dataset_rejects_bad_input(tmp_path, text):
    with pytest.raises(ValueError):
        ev.load_dataset(_write(tmp_path, "bad.jsonl", text))


def test_select_metric_names_defaults_depend_on_reference():
    assert "context_recall" in ev.select_metric_names(None, True)
    assert "context_recall" not in ev.select_metric_names(None, False)


def test_select_metric_names_validation():
    with pytest.raises(ValueError):
        ev.select_metric_names(["bogus"], True)
    with pytest.raises(ValueError):
        ev.select_metric_names(["context_recall"], False)
    assert ev.select_metric_names(["faithfulness", "faithfulness"], False) == ["faithfulness"]


def test_summarize_ignores_none_and_nan_and_counts_failures():
    rows = [{"f": 1.0}, {"f": 0.5}, {"f": None}, {"f": float("nan")}]
    s = ev.summarize(rows, ["f"])["f"]
    assert s == {"mean": 0.75, "scored": 2, "failed": 2}
    assert ev.summarize([{"f": None}], ["f"])["f"]["mean"] is None


def test_latency_stats():
    assert ev.latency_stats([]) is None
    st = ev.latency_stats([100, 200, 300, 400])
    assert st["mean_ms"] == 250.0 and st["p95_ms"] == 400.0


def test_write_report_creates_json_and_csv(tmp_path):
    report = {"rows": [{"question": "q", "retrieved_contexts": ["a"], "faithfulness": 0.9}]}
    path = ev.write_report(report, tmp_path / "out", "my tag/1")
    assert path.exists() and path.with_suffix(".csv").exists()
    assert "my-tag-1" in path.name
    assert json.loads(path.read_text(encoding="utf-8"))["rows"][0]["faithfulness"] == 0.9
