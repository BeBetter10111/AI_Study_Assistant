"""Exam-prep mode: syllabus -> exam, server-side grading, weak-topic report, isolation, validation."""
import json

import pytest

import app.api.endpoints.exams as exams_module
from app.api.security.rate_limiter import limiter
from app.exams import logic

PW = "firstpassword1"


async def _user(client, name):
    res = await client.post("/api/auth/register", json={"username": name, "email": f"{name}@example.com", "password": PW})
    assert res.status_code == 201, res.text
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


def _fake_generator(skip=()):
    """Stand-in for the retrieval + LLM step: deterministic MCQs per topic (topics in `skip` have no content)."""
    def gen(topic, count, document_id, owner_id):
        if topic in skip:
            return []
        return [{"question": f"{topic} - câu {i + 1}?", "options": ["A", "B", "C", "D"],
                 "correct_index": i % 4, "explanation": f"Giải thích {topic} {i + 1}"} for i in range(count)]
    return gen


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(limiter, "enabled", False)
    monkeypatch.setattr(exams_module, "_generate_topic_questions", _fake_generator())


SYLLABUS = "1. Stack\n2. Queue\n3. Linked list"


async def _make_exam(client, headers, n=6, **extra):
    res = await client.post("/api/exams/generate", headers=headers,
                            json={"syllabus": SYLLABUS, "num_questions": n, "title": "Giữa kỳ", **extra})
    assert res.status_code == 200, res.text
    return res.json()


# ---- pure logic ---------------------------------------------------------------------------------------------
def test_syllabus_parsing_allocation_and_grading_logic():
    assert logic.parse_syllabus("1. Stack\n- Queue\n• Cây; Đồ thị\nstack") == ["Stack", "Queue", "Cây", "Đồ thị"]
    assert logic.allocate_questions(7, ["a", "b", "c"]) == [("a", 3), ("b", 2), ("c", 2)]
    assert logic.allocate_questions(2, ["a", "b", "c"]) == [("a", 1), ("b", 1)]
    qs = [{"index": 0, "topic": "A", "correct_index": 1}, {"index": 1, "topic": "A", "correct_index": 0},
          {"index": 2, "topic": "B", "correct_index": 2}]
    g = logic.grade_attempt(qs, {0: 1, 1: 3, 2: None})
    assert (g["correct"], g["unanswered"], g["score"]) == (1, 1, 33.3)
    assert g["per_topic"] == {"A": {"correct": 1, "total": 2}, "B": {"correct": 0, "total": 1}}


def test_topic_report_uses_smoothing_and_minimum_evidence():
    rows = logic.topic_report([{"A": {"correct": 1, "total": 1}, "B": {"correct": 1, "total": 5}}], 0.6, 3)
    by = {r["topic"]: r for r in rows}
    assert by["A"]["needs_more_data"] and not by["A"]["weak"]          # one lucky answer is neither strong nor weak
    assert by["B"]["weak"] and by["B"]["smoothed_accuracy"] == round(2 / 7, 3)
    assert rows[0]["topic"] == "B"                                      # weakest first
    lo, hi = logic.wilson_interval(8, 10)
    assert 0.49 < lo < 0.5 and 0.94 < hi < 0.96 and logic.wilson_interval(0, 0) == (0.0, 1.0)


# ---- API ----------------------------------------------------------------------------------------------------
async def test_generate_hides_answers_and_allocates_questions_evenly(app_client):
    headers = await _user(app_client, "ex1")
    exam = await _make_exam(app_client, headers, 7)
    assert exam["num_questions"] == 7 and exam["topics"] == ["Stack", "Queue", "Linked list"]
    assert "correct_index" not in json.dumps(exam) and "explanation" not in json.dumps(exam)
    counts = {t: sum(q["topic"] == t for q in exam["questions"]) for t in exam["topics"]}
    assert counts == {"Stack": 3, "Queue": 2, "Linked list": 2}
    again = (await app_client.get(f"/api/exams/{exam['id']}", headers=headers)).json()
    assert "correct_index" not in json.dumps(again)


async def test_topics_without_content_are_skipped_and_all_empty_is_404(app_client, monkeypatch):
    headers = await _user(app_client, "ex2")
    monkeypatch.setattr(exams_module, "_generate_topic_questions", _fake_generator(skip=("Queue",)))
    exam = await _make_exam(app_client, headers, 6)
    assert exam["skipped_topics"] == ["Queue"] and "Queue" not in exam["topics"]
    monkeypatch.setattr(exams_module, "_generate_topic_questions", _fake_generator(skip=("Stack", "Queue", "Linked list")))
    res = await app_client.post("/api/exams/generate", headers=headers, json={"syllabus": SYLLABUS, "num_questions": 5})
    assert res.status_code == 404


async def test_submit_grades_on_server_and_reports_weak_topics(app_client):
    headers = await _user(app_client, "ex3")
    exam = await _make_exam(app_client, headers, 6)
    # correct_index of question i is i % 4 in the fake generator (per topic, restarting at 0)
    by_topic = {}
    for q in exam["questions"]:
        by_topic.setdefault(q["topic"], []).append(q)
    answers = []
    for topic, qs in by_topic.items():
        for pos, q in enumerate(qs):
            answers.append({"index": q["index"], "choice": pos % 4 if topic == "Stack" else (pos + 1) % 4})
    res = await app_client.post(f"/api/exams/{exam['id']}/submit", headers=headers,
                                json={"answers": answers, "duration_s": 120})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["correct"] == 2 and body["total"] == 6 and body["score"] == 33.3     # only Stack is right
    weak = {t["topic"]: t["weak"] for t in body["per_topic"]}
    assert weak == {"Stack": False, "Queue": True, "Linked list": True}
    assert body["review"][0]["correct_index"] is not None and "explanation" in body["review"][0]


async def test_unanswered_counts_as_wrong_and_attempts_are_listed(app_client):
    headers = await _user(app_client, "ex4")
    exam = await _make_exam(app_client, headers, 3)
    res = await app_client.post(f"/api/exams/{exam['id']}/submit", headers=headers, json={"answers": []})
    assert res.json()["score"] == 0 and res.json()["unanswered"] == 3
    listing = (await app_client.get("/api/exams", headers=headers)).json()
    assert listing[0]["attempts"] == 1 and listing[0]["last_score"] == 0


async def test_submit_validation(app_client):
    headers = await _user(app_client, "ex5")
    exam = await _make_exam(app_client, headers, 3)
    url = f"/api/exams/{exam['id']}/submit"
    assert (await app_client.post(url, headers=headers, json={"answers": [{"index": 99, "choice": 0}]})).status_code == 400
    assert (await app_client.post(url, headers=headers, json={"answers": [{"index": 0, "choice": 9}]})).status_code == 400
    dup = {"answers": [{"index": 0, "choice": 0}, {"index": 0, "choice": 1}]}
    assert (await app_client.post(url, headers=headers, json=dup)).status_code == 400
    assert (await app_client.post(url, headers=headers, json={"answers": [{"index": 0, "choice": -1}]})).status_code == 422


async def test_weak_topic_report_aggregates_attempts_across_exams(app_client):
    headers = await _user(app_client, "ex6")
    for _ in range(2):
        exam = await _make_exam(app_client, headers, 6)
        wrong = [{"index": q["index"], "choice": 3} for q in exam["questions"]]       # mostly wrong
        await app_client.post(f"/api/exams/{exam['id']}/submit", headers=headers, json={"answers": wrong})
    report = (await app_client.get("/api/exams/report/weak-topics", headers=headers)).json()
    assert report["attempts_considered"] == 2
    assert report["topics"][0]["total"] == 4 and report["weak_topics"]
    assert report["recommendations"][0]["action"] == "quiz"


async def test_exams_are_private_to_their_owner_and_deletable(app_client):
    a = await _user(app_client, "ex7a")
    b = await _user(app_client, "ex7b")
    exam = await _make_exam(app_client, a, 3)
    assert (await app_client.get(f"/api/exams/{exam['id']}", headers=b)).status_code == 404
    assert (await app_client.post(f"/api/exams/{exam['id']}/submit", headers=b, json={"answers": []})).status_code == 404
    assert (await app_client.get("/api/exams/report/weak-topics", headers=b)).json()["attempts_considered"] == 0
    assert (await app_client.delete(f"/api/exams/{exam['id']}", headers=a)).status_code == 204
    assert (await app_client.get(f"/api/exams/{exam['id']}", headers=a)).status_code == 404


async def test_request_validation_and_limits(app_client, monkeypatch):
    headers = await _user(app_client, "ex8")
    assert (await app_client.post("/api/exams/generate", headers=headers, json={"syllabus": "ab"})).status_code == 422
    assert (await app_client.post("/api/exams/generate", headers=headers, json={"syllabus": "  \n -- ", "num_questions": 3})).status_code in (400, 422)
    bad_doc = {"syllabus": SYLLABUS, "document_id": "not-a-uuid"}
    assert (await app_client.post("/api/exams/generate", headers=headers, json=bad_doc)).status_code == 400
    monkeypatch.setattr(exams_module.settings, "exam_max_questions", 4)
    assert (await _make_exam(app_client, headers, 30))["num_questions"] == 4        # capped by server setting
    monkeypatch.setattr(exams_module.settings, "max_exams_per_user", 1)
    assert (await app_client.post("/api/exams/generate", headers=headers, json={"syllabus": SYLLABUS})).status_code == 409
