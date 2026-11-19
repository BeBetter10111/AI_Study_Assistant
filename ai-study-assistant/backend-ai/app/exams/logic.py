"""Pure logic for exam-prep mode and lecturer mode (no DB / LLM access, so it is easy to unit-test).

* syllabus parsing and question allocation
* server-side grading and per-topic statistics
* weak-topic estimation with a Beta prior and a Wilson interval
* rubric normalisation, paper assembly (stratified by difficulty) and export (Markdown / Word)
"""
from __future__ import annotations

import io
import math
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

DIFFICULTIES = ("easy", "medium", "hard")
DEFAULT_MIX = {"easy": 0.3, "medium": 0.5, "hard": 0.2}
OPTION_LETTERS = "ABCDEFGH"

# Syllabus
_LEAD_RE = re.compile(r"^\s*(?:[-*•–—]+|\d{1,2}(?:\.\d{1,2})*[.)]?|[a-zA-Z][.)])\s+")


def parse_syllabus(text: str, max_topics: int = 12) -> List[str]:
    """Topics of a syllabus: one per line / bullet / semicolon; duplicates and tiny items removed."""
    parts = re.split(r"[\n;]+", text or "")
    if len(parts) == 1 and len(parts[0]) > 80 and parts[0].count(",") >= 2:
        parts = parts[0].split(",")          # "A, B, C, D" typed on one line
    topics: List[str] = []
    seen = set()
    for raw in parts:
        topic = _LEAD_RE.sub("", raw).strip(" .:-–—\t")
        topic = re.sub(r"\s+", " ", topic)[:200]
        key = topic.lower()
        if len(topic) < 3 or key in seen:
            continue
        seen.add(key)
        topics.append(topic)
        if len(topics) >= max_topics:
            break
    return topics


def allocate_questions(total: int, topics: Sequence[str]) -> List[Tuple[str, int]]:
    """Split ``total`` questions across topics as evenly as possible (earlier topics get the remainder).

    With fewer questions than topics only the first ``total`` topics get one question each.
    """
    if total <= 0 or not topics:
        return []
    base, extra = divmod(total, len(topics))
    out = [(t, base + (1 if i < extra else 0)) for i, t in enumerate(topics)]
    return [(t, n) for t, n in out if n > 0]


def clean_mcq(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate one LLM-generated multiple-choice item; ``None`` if unusable."""
    question = str(raw.get("question") or "").strip()
    options = [str(o).strip() for o in (raw.get("options") or []) if str(o).strip()]
    idx = raw.get("correct_index")
    if not question or len(options) < 2 or len(set(o.lower() for o in options)) != len(options):
        return None
    if not isinstance(idx, int) or isinstance(idx, bool) or not (0 <= idx < len(options)):
        return None
    return {"question": question, "options": options[:8], "correct_index": idx,
            "explanation": str(raw.get("explanation") or "").strip()}


# Grading and statistics
def grade_attempt(questions: Sequence[Dict[str, Any]], answers: Dict[int, Optional[int]]) -> Dict[str, Any]:
    """Grade an exam attempt on the server (the answer key never leaves the server before submission).

    ``questions``: [{index, topic, question, options, correct_index, explanation}]
    ``answers``  : {question index: chosen option index or None}
    """
    correct = unanswered = 0
    per_topic: Dict[str, Dict[str, int]] = {}
    review: List[Dict[str, Any]] = []
    for q in questions:
        choice = answers.get(q["index"])
        is_correct = choice is not None and choice == q["correct_index"]
        correct += int(is_correct)
        unanswered += int(choice is None)
        bucket = per_topic.setdefault(q["topic"], {"correct": 0, "total": 0})
        bucket["total"] += 1
        bucket["correct"] += int(is_correct)
        review.append({**q, "your_choice": choice, "is_correct": is_correct})
    total = len(questions)
    return {
        "correct": correct, "total": total, "unanswered": unanswered,
        "score": round(100.0 * correct / total, 1) if total else 0.0,
        "per_topic": per_topic, "review": review,
    }


def wilson_interval(correct: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion (well behaved for the small n we have per topic)."""
    if n <= 0:
        return 0.0, 1.0
    p = correct / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom)


def topic_report(per_topic_attempts: Sequence[Dict[str, Dict[str, int]]],
                 weak_threshold: float = 0.6, min_questions: int = 3) -> List[Dict[str, Any]]:
    """Aggregate per-topic results over many attempts and rank topics from weakest to strongest.

    Smoothed accuracy uses a Beta(1,1) prior (Laplace):  p~ = (c + 1) / (n + 2),  so a topic with a single
    lucky answer is not shown as "mastered".  A topic is **weak** when n >= min_questions and p~ < threshold.
    """
    agg: Dict[str, Dict[str, int]] = {}
    last: Dict[str, float] = {}
    for attempt in per_topic_attempts:                      # oldest -> newest
        for topic, res in attempt.items():
            a = agg.setdefault(topic, {"correct": 0, "total": 0})
            a["correct"] += int(res.get("correct", 0))
            a["total"] += int(res.get("total", 0))
            if res.get("total"):
                last[topic] = res["correct"] / res["total"]
    rows = []
    for topic, a in agg.items():
        c, n = a["correct"], a["total"]
        if n <= 0:
            continue
        smoothed = (c + 1) / (n + 2)
        low, high = wilson_interval(c, n)
        enough = n >= min_questions
        rows.append({
            "topic": topic, "correct": c, "total": n,
            "accuracy": round(c / n, 3), "smoothed_accuracy": round(smoothed, 3),
            "wilson_low": round(low, 3), "wilson_high": round(high, 3),
            "last_accuracy": round(last.get(topic, c / n), 3),
            "weak": bool(enough and smoothed < weak_threshold),
            "needs_more_data": not enough,
        })
    rows.sort(key=lambda r: (r["smoothed_accuracy"], -r["total"], r["topic"]))
    return rows


# Rubrics
def normalize_rubric(criteria: Sequence[Dict[str, Any]], total_points: float = 10.0) -> List[Dict[str, Any]]:
    """Clean LLM rubric criteria and rescale them so the points add up to exactly ``total_points``."""
    clean = []
    for c in criteria:
        name = str(c.get("criterion") or "").strip()
        try:
            pts = float(c.get("max_points"))
        except (TypeError, ValueError):
            continue
        if not name or not math.isfinite(pts) or pts <= 0:
            continue
        clean.append({
            "criterion": name[:200], "max_points": pts,
            "full": str(c.get("full") or "").strip(), "partial": str(c.get("partial") or "").strip(),
            "none": str(c.get("none") or "").strip(),
        })
    clean = clean[:6]
    if not clean or total_points <= 0:
        return []
    raw_sum = sum(c["max_points"] for c in clean)
    for c in clean:
        c["max_points"] = max(0.5, round(c["max_points"] / raw_sum * total_points * 2) / 2)   # steps of 0.5
    drift = round(total_points - sum(c["max_points"] for c in clean), 2)
    if drift:
        biggest = max(clean, key=lambda c: c["max_points"])
        biggest["max_points"] = round(max(0.5, biggest["max_points"] + drift), 2)
    return clean


def clamp_grade(rubric: Sequence[Dict[str, Any]], scores: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge LLM grading into the rubric: scores are clamped to [0, max] and the total is recomputed."""
    by_name = {str(s.get("criterion", "")).strip().lower(): s for s in scores}
    rows, total = [], 0.0
    for c in rubric:
        s = by_name.get(c["criterion"].lower(), {})
        try:
            pts = float(s.get("points", 0))
        except (TypeError, ValueError):
            pts = 0.0
        if not math.isfinite(pts):
            pts = 0.0
        pts = round(min(max(pts, 0.0), c["max_points"]) * 2) / 2
        total += pts
        rows.append({"criterion": c["criterion"], "points": pts, "max_points": c["max_points"],
                     "justification": str(s.get("justification") or "").strip()[:500]})
    return {"criteria": rows, "total": round(total, 2), "max_total": round(sum(c["max_points"] for c in rubric), 2)}


# Paper assembly
def select_items(items: Sequence[Dict[str, Any]], counts: Dict[str, int],
                 mix: Optional[Dict[str, float]] = None, rng: Optional[random.Random] = None
                 ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Pick items per question type, stratified by difficulty. Returns (selected, shortfall per type)."""
    rng = rng or random.Random()
    mix = mix or DEFAULT_MIX
    selected: List[Dict[str, Any]] = []
    shortfall: Dict[str, int] = {}
    for qtype, want in counts.items():
        if want <= 0:
            continue
        pool = {d: [i for i in items if i["qtype"] == qtype and i["difficulty"] == d] for d in DIFFICULTIES}
        for lst in pool.values():
            rng.shuffle(lst)
        # largest-remainder allocation of `want` over difficulties
        raw = {d: want * mix.get(d, 0.0) for d in DIFFICULTIES}
        target = {d: int(raw[d]) for d in DIFFICULTIES}
        for d in sorted(DIFFICULTIES, key=lambda d: raw[d] - target[d], reverse=True):
            if sum(target.values()) >= want:
                break
            target[d] += 1
        taken: List[Dict[str, Any]] = []
        for d in DIFFICULTIES:
            taken += [pool[d].pop() for _ in range(min(target[d], len(pool[d])))]
        leftovers = [i for d in DIFFICULTIES for i in pool[d]]
        rng.shuffle(leftovers)
        while len(taken) < want and leftovers:
            taken.append(leftovers.pop())
        if len(taken) < want:
            shortfall[qtype] = want - len(taken)
        selected += taken
    return selected, shortfall


def shuffle_item(item: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Shuffle MCQ options and remap ``correct_index`` (open questions are returned unchanged)."""
    if item.get("qtype") != "mcq" or not item.get("options"):
        return dict(item)
    order = list(range(len(item["options"])))
    rng.shuffle(order)
    out = dict(item)
    out["options"] = [item["options"][i] for i in order]
    if item.get("correct_index") is not None:
        out["correct_index"] = order.index(item["correct_index"])
    return out


def item_points(item: Dict[str, Any], mcq_points: float) -> float:
    if item.get("qtype") == "mcq":
        return float(mcq_points)
    return float(sum(c["max_points"] for c in (item.get("rubric") or []))) or 0.0


def build_paper(title: str, items: Sequence[Dict[str, Any]], mcq_points: float = 1.0, shuffle: bool = False,
                seed: Optional[int] = None) -> Dict[str, Any]:
    """Snapshot of an exam paper: MCQ section first, then short answer, then essay."""
    rng = random.Random(seed)
    order = {"mcq": 0, "short_answer": 1, "essay": 2}
    ordered = sorted(items, key=lambda i: order.get(i["qtype"], 9))
    questions = []
    for n, item in enumerate(ordered, start=1):
        q = shuffle_item(item, rng) if shuffle else dict(item)
        q["number"] = n
        q["points"] = item_points(q, mcq_points)
        questions.append(q)
    return {"title": title, "questions": questions, "total_points": round(sum(q["points"] for q in questions), 2)}


_SECTION_TITLES = {"mcq": "Phần I. Trắc nghiệm", "short_answer": "Phần II. Câu hỏi ngắn", "essay": "Phần III. Tự luận"}


def _fmt_points(v: float) -> str:
    return f"{v:g}"


def paper_markdown(paper: Dict[str, Any], include_answers: bool = False) -> str:
    lines = [f"# {paper['title']}", "", f"Tổng điểm: {_fmt_points(paper['total_points'])}", ""]
    current = None
    for q in paper["questions"]:
        if q["qtype"] != current:
            current = q["qtype"]
            lines += [f"## {_SECTION_TITLES.get(current, current)}", ""]
        lines.append(f"**Câu {q['number']}** ({_fmt_points(q['points'])} điểm). {q['question']}")
        for i, opt in enumerate(q.get("options") or []):
            lines.append(f"- {OPTION_LETTERS[i]}. {opt}")
        lines.append("")
    if include_answers:
        lines += ["---", "", "## Đáp án và thang điểm", ""]
        for q in paper["questions"]:
            if q["qtype"] == "mcq":
                ci = q.get("correct_index")
                ans = OPTION_LETTERS[ci] if isinstance(ci, int) and 0 <= ci < len(OPTION_LETTERS) else "?"
                lines.append(f"- Câu {q['number']}: {ans}")
            else:
                lines.append(f"- Câu {q['number']}: {q.get('answer_key', '')}")
                for c in q.get("rubric") or []:
                    lines.append(f"  - {c['criterion']} ({_fmt_points(c['max_points'])} điểm)")
        lines.append("")
    return "\n".join(lines)


def paper_docx(paper: Dict[str, Any], include_answers: bool = False) -> bytes:
    """Word export (python-docx). Section order: header, questions, optional answer key + rubric."""
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    doc.styles["Normal"].font.name = "Times New Roman"
    doc.styles["Normal"].font.size = Pt(12)
    doc.add_heading(paper["title"], level=1)
    doc.add_paragraph("Họ và tên: ................................    Mã số: ....................")
    doc.add_paragraph(f"Tổng điểm: {_fmt_points(paper['total_points'])}")
    current = None
    for q in paper["questions"]:
        if q["qtype"] != current:
            current = q["qtype"]
            doc.add_heading(_SECTION_TITLES.get(current, current), level=2)
        p = doc.add_paragraph()
        p.add_run(f"Câu {q['number']} ({_fmt_points(q['points'])} điểm). ").bold = True
        p.add_run(q["question"])
        for i, opt in enumerate(q.get("options") or []):
            doc.add_paragraph(f"{OPTION_LETTERS[i]}. {opt}")
        if q["qtype"] != "mcq":
            doc.add_paragraph("")
    if include_answers:
        doc.add_page_break()
        doc.add_heading("Đáp án và thang điểm", level=1)
        for q in paper["questions"]:
            if q["qtype"] == "mcq":
                ci = q.get("correct_index")
                ans = OPTION_LETTERS[ci] if isinstance(ci, int) and 0 <= ci < len(OPTION_LETTERS) else "?"
                doc.add_paragraph(f"Câu {q['number']}: {ans}")
                continue
            doc.add_paragraph(f"Câu {q['number']}: {q.get('answer_key', '')}")
            rubric = q.get("rubric") or []
            if rubric:
                table = doc.add_table(rows=1, cols=4)
                table.style = "Table Grid"
                for cell, text in zip(table.rows[0].cells, ("Tiêu chí", "Điểm", "Đạt tối đa", "Một phần")):
                    cell.text = text
                for c in rubric:
                    row = table.add_row().cells
                    row[0].text, row[1].text = c["criterion"], _fmt_points(c["max_points"])
                    row[2].text, row[3].text = c.get("full", ""), c.get("partial", "")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
