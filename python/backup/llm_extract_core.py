# llm_extract_core.py (patch)

from __future__ import annotations
import json
import re
from typing import List, Dict, Optional, Any
from ollama import chat
from difflib import SequenceMatcher


SYSTEM_PROMPT = """You are an information extraction assistant.

You will receive:
- A context note
- The full document text (or a chunk)
- A JSON list of questions with ids

Return ONLY valid JSON, and ONLY this structure:
{
  "answers": {
    "<id>": <string or null>,
    ...
  }
}

Rules:
- Use only evidence from the document text.
- If not present / not inferable, use null.
- Do not add extra keys.
- No markdown, no explanation.
"""


EVIDENCE_SYSTEM_PROMPT = """You are an information extraction assistant.

You will receive:
- A context note
- The full document text (or a chunk)
- A JSON list of questions with ids

Return ONLY valid JSON, and ONLY this structure:
{
  "answers": {
    "<id>": {
      "value": <string or null>,
      "confidence": <number between 0 and 1 or null>,
      "excerpt_location": {
        "page": <integer or null>,
        "sentence": <string or null>
      }
    }
  }
}

Rules:
- Use only evidence from the document text.
- If not present / not inferable, use null for value.
- page must come from explicit page markers in the document text, such as:
  ===== PAGE 7 =====
- sentence must be the shortest supporting sentence copied from the document text.
- Do not add extra keys.
- No markdown, no explanation.
"""


_PAGE_MARKER_RE = re.compile(
    r"^=+\s*PAGE\s+(\d+)\b[^\n]*=+\s*$",
    flags=re.MULTILINE | re.IGNORECASE,
)


def _as_int(v: Any) -> int | None:
    try:
        if v is None or str(v).strip() == "":
            return None
        return int(float(str(v).strip()))
    except Exception:
        return None


def _as_float(v: Any) -> float | None:
    try:
        if v is None or str(v).strip() == "":
            return None
        x = float(str(v).strip())
        if x < 0:
            x = 0.0
        if x > 1:
            x = 1.0
        return x
    except Exception:
        return None


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _norm_match(s: str) -> str:
    s = _norm_ws(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _page_segments(text: str) -> list[dict[str, Any]]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    matches = list(_PAGE_MARKER_RE.finditer(text))
    if not matches:
        return [{"page": None, "text": text}]

    out: list[dict[str, Any]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append({
            "page": int(m.group(1)),
            "text": text[start:end].strip(),
        })
    return out


def _candidate_sentences(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    for raw_line in re.split(r"[\r\n]+", text or ""):
        line = _norm_ws(raw_line)
        if not line:
            continue

        parts = [line] + re.split(r"(?<=[.!?])\s+", line)
        for part in parts:
            cand = _norm_ws(part)
            if not cand:
                continue
            key = _norm_match(cand)
            if key and key not in seen:
                seen.add(key)
                out.append(cand)

    return out


def _locate_excerpt(document_text: str, sentence: str | None, hinted_page: int | None) -> dict[str, Any]:
    sent = _norm_ws(sentence or "")
    if not sent:
        return {"page": hinted_page, "sentence": None}

    segs = _page_segments(document_text)
    if hinted_page is not None:
        segs = sorted(segs, key=lambda d: 0 if d.get("page") == hinted_page else 1)

    target = _norm_match(sent)

    # exact normalized match
    for seg in segs:
        for cand in _candidate_sentences(seg.get("text", "")):
            if _norm_match(cand) == target:
                return {"page": seg.get("page"), "sentence": cand}

    # containment / near-exact
    best_soft: tuple[float, int | None, str] | None = None
    for seg in segs:
        for cand in _candidate_sentences(seg.get("text", "")):
            ck = _norm_match(cand)
            if not ck or not target:
                continue
            if target in ck or ck in target:
                score = min(len(target), len(ck)) / max(len(target), len(ck))
                if best_soft is None or score > best_soft[0]:
                    best_soft = (score, seg.get("page"), cand)

    if best_soft and best_soft[0] >= 0.88:
        return {"page": best_soft[1], "sentence": best_soft[2]}

    # fuzzy fallback
    best_fuzzy: tuple[float, int | None, str] | None = None
    for seg in segs:
        for cand in _candidate_sentences(seg.get("text", "")):
            ratio = SequenceMatcher(None, target, _norm_match(cand)).ratio()
            if best_fuzzy is None or ratio > best_fuzzy[0]:
                best_fuzzy = (ratio, seg.get("page"), cand)

    if best_fuzzy and best_fuzzy[0] >= 0.82:
        return {"page": best_fuzzy[1], "sentence": best_fuzzy[2]}

    # keep what model gave, but page may still be useful
    return {"page": hinted_page, "sentence": sent}


def _coerce_rich_answer(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        value = raw.get("value", raw.get("answer"))
        confidence = _as_float(raw.get("confidence"))

        loc = raw.get("excerpt_location") if isinstance(raw.get("excerpt_location"), dict) else {}
        page = _as_int(loc.get("page") if loc else raw.get("page"))
        sentence = (
            loc.get("sentence")
            if loc
            else raw.get("sentence", raw.get("excerpt"))
        )
    else:
        value = raw
        confidence = None
        page = None
        sentence = None

    value = None if value is None or not str(value).strip() else str(value).strip()

    return {
        "value": value,
        "confidence": confidence,
        "excerpt_location": {
            "page": page,
            "sentence": _norm_ws(str(sentence)) if sentence else None,
        },
    }


def _rich_answer_score(item: dict[str, Any]) -> float:
    if not isinstance(item, dict):
        return 0.0
    value = item.get("value")
    if value is None or not str(value).strip():
        return 0.0

    conf = _as_float(item.get("confidence")) or 0.0
    loc = item.get("excerpt_location") if isinstance(item.get("excerpt_location"), dict) else {}
    page_bonus = 0.05 if loc.get("page") is not None else 0.0
    sent_bonus = 0.05 if loc.get("sentence") else 0.0
    len_bonus = min(len(str(value).strip()), 60) / 1000.0
    return conf + page_bonus + sent_bonus + len_bonus


def extract_json_object(text: str) -> Optional[str]:
    """Pull a JSON object out of markdown/codefences/prose."""
    if not text:
        return None

    # 1) Code fence: ```json { ... } ```
    m = re.search(r"```(?:json)?\s*({.*?})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 2) First balanced {...} block, respecting strings/escapes
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        # not in string
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1].strip()

    return None


def safe_load_json_from_model(text: str) -> Dict[str, Any]:
    snippet = extract_json_object(text)
    if not snippet:
        return {}
    try:
        obj = json.loads(snippet)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _repair_to_json(model: str, bad_text: str) -> Dict[str, Any]:
    repair_system = "You output ONLY valid JSON. No markdown. No extra text."
    resp = chat(
        model=model,
        messages=[
            {"role": "system", "content": repair_system},
            {"role": "user", "content": f"Fix this into valid JSON only:\n\n{bad_text}"},
        ],
        options={"temperature": 0.0},
    )
    content = getattr(getattr(resp, "message", None), "content", None) or ""
    parsed = safe_load_json_from_model(content)
    return parsed if isinstance(parsed, dict) else {}


def _normalize_question_for_model(q: str) -> str:
    # keeps user-visible key unchanged elsewhere; this is only for the model prompt
    q2 = (q or "").strip()
    q2 = re.sub(r"\s+", " ", q2)
    q2 = q2.rstrip(":").strip()
    return q2


def answer_questions_json(
    *,
    model: str,
    document_text: str,
    questions: List[str],
    context_note: str = "The information might be spread out through the entire document.",
    temperature: float = 0.1,
) -> Dict[str, Optional[str]]:
    orig_questions = [q.strip() for q in questions if q and q.strip()]
    if not orig_questions:
        return {}

    # Build an ID map so punctuation (like trailing :) never breaks key matching.
    q_items = []
    id_to_key: Dict[str, str] = {}
    for i, q in enumerate(orig_questions, start=1):
        qid = f"q{i}"
        id_to_key[qid] = q
        q_items.append({"id": qid, "question": _normalize_question_for_model(q)})

    user_prompt = (
        f"Context note:\n{context_note}\n\n"
        f"Questions:\n{json.dumps(q_items, ensure_ascii=False, indent=2)}\n\n"
        "Document text:\n<doc>\n"
        f"{document_text}\n"
        "</doc>\n"
        "\nReturn JSON now."
    )

    resp = chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": temperature},
    )
    print(resp)
    content = getattr(getattr(resp, "message", None), "content", None) or ""
    parsed = safe_load_json_from_model(content)
    if not isinstance(parsed, dict):
        parsed = _repair_to_json(model, content)

    # --- flexible answer extraction ---
    answers = None

    if isinstance(parsed, dict):
        # Preferred: {"answers": {"q1": "..."}}
        if isinstance(parsed.get("answers"), dict):
            answers = parsed["answers"]

        # Common: {"q1": "..."} (no wrapper)
        elif any(k.startswith("q") and k[1:].isdigit() for k in parsed.keys()):
            answers = parsed

        # Also common: keys are the human questions themselves
        else:
            answers = {}

    if not isinstance(answers, dict):
        answers = {}


    # Remap back to the original question strings as keys.
    out: Dict[str, Optional[str]] = {}
    for qid, key in id_to_key.items():
        v = None

        # Preferred: id-based
        if isinstance(answers, dict):
            v = answers.get(qid, None)

            # Fallback: original question string
            if v is None:
                orig_key = id_to_key[qid]
                v = answers.get(orig_key, None)

            # Fallback: stripped colon
            if v is None:
                v = answers.get(orig_key.rstrip(":").strip(), None)

        # Normalize final value
        if v is None:
            out[key] = None
        else:
            out[key] = str(v).strip() if str(v).strip() else None
    return out


def _chunk_text(s: str, max_chars: int = 12000, overlap: int = 800) -> list[str]:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    page_segs = _page_segments(s)

    # No page markers: original char-based chunking
    if len(page_segs) == 1 and page_segs[0].get("page") is None:
        chunks = []
        i = 0
        while i < len(s):
            j = min(len(s), i + max_chars)
            chunks.append(s[i:j])
            if j == len(s):
                break
            i = max(0, j - overlap)
        return chunks

    # Page-aware chunking
    page_blocks: list[str] = []
    for seg in page_segs:
        page = seg.get("page")
        txt = seg.get("text", "")
        page_blocks.append(f"===== PAGE {page} =====\n{txt}".strip())

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for block in page_blocks:
        block_len = len(block) + 2
        if current and current_len + block_len > max_chars:
            chunks.append("\n\n".join(current))

            if overlap > 0:
                carry: list[str] = []
                carry_len = 0
                for old in reversed(current):
                    carry.insert(0, old)
                    carry_len += len(old) + 2
                    if carry_len >= overlap:
                        break
                current = carry
                current_len = sum(len(x) + 2 for x in current)
            else:
                current = []
                current_len = 0

        current.append(block)
        current_len += block_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks

def answer_questions_json_chunked(
    *,
    model: str,
    document_text: str,
    questions: List[str],
    context_note: str = "The information might be spread out through the entire document.",
    max_chars: int = 12000,
    overlap: int = 800,
) -> Dict[str, Optional[str]]:
    qs = [q.strip() for q in questions if q and q.strip()]
    if not qs:
        return {}

    merged: Dict[str, Optional[str]] = {q: None for q in qs}
    chunks = _chunk_text(document_text, max_chars=max_chars, overlap=overlap)

    for idx, ch in enumerate(chunks, start=1):
        partial = answer_questions_json(
            model=model,
            document_text=f"[CHUNK {idx}/{len(chunks)}]\n{ch}",
            questions=qs,
            context_note=context_note + " Answer only if the information is present in this chunk.",
            temperature=0.1,
        )
        for q in qs:
            new_v = partial.get(q)
            old_v = merged.get(q)

            def _score(v: Optional[str]) -> int:
                if not v:
                    return 0
                return len(v.strip())

            if _score(new_v) > _score(old_v):
                merged[q] = new_v

    return merged

def answer_questions_json_evidence(
    *,
    model: str,
    document_text: str,
    questions: List[str],
    context_note: str = "The information might be spread out through the entire document.",
    temperature: float = 0.1,
) -> Dict[str, Dict[str, Any]]:
    orig_questions = [q.strip() for q in questions if q and q.strip()]
    if not orig_questions:
        return {}

    q_items = []
    id_to_key: Dict[str, str] = {}
    for i, q in enumerate(orig_questions, start=1):
        qid = f"q{i}"
        id_to_key[qid] = q
        q_items.append({"id": qid, "question": _normalize_question_for_model(q)})

    user_prompt = (
        f"Context note:\n{context_note}\n\n"
        f"Questions:\n{json.dumps(q_items, ensure_ascii=False, indent=2)}\n\n"
        "Document text:\n<doc>\n"
        f"{document_text}\n"
        "</doc>\n\n"
        "Return JSON now."
    )

    resp = chat(
        model=model,
        messages=[
            {"role": "system", "content": EVIDENCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": temperature},
    )

    content = getattr(getattr(resp, "message", None), "content", None) or ""
    parsed = safe_load_json_from_model(content)
    if not isinstance(parsed, dict) or not parsed:
        parsed = _repair_to_json(model, content)

    answers: dict[str, Any] = {}
    if isinstance(parsed, dict):
        if isinstance(parsed.get("answers"), dict):
            answers = parsed["answers"]
        elif any(k.startswith("q") and k[1:].isdigit() for k in parsed.keys()):
            answers = parsed

    out: Dict[str, Dict[str, Any]] = {}
    for qid, key in id_to_key.items():
        raw = answers.get(qid)
        if raw is None:
            raw = answers.get(key)
        if raw is None:
            raw = answers.get(key.rstrip(":").strip())

        item = _coerce_rich_answer(raw)

        if item["value"] is None:
            out[key] = {
                "value": None,
                "confidence": 0.0,
                "excerpt_location": {"page": None, "sentence": None},
            }
            continue

        hinted_page = item.get("excerpt_location", {}).get("page")
        hinted_sentence = item.get("excerpt_location", {}).get("sentence")
        verified_loc = _locate_excerpt(document_text, hinted_sentence, hinted_page)

        conf = item.get("confidence")
        if conf is None:
            conf = 0.9 if verified_loc.get("sentence") else 0.55

        out[key] = {
            "value": item["value"],
            "confidence": round(max(0.0, min(1.0, conf)), 3),
            "excerpt_location": {
                "page": verified_loc.get("page"),
                "sentence": verified_loc.get("sentence"),
            },
        }

    return out


def answer_questions_json_chunked_evidence(
    *,
    model: str,
    document_text: str,
    questions: List[str],
    context_note: str = "The information might be spread out through the entire document.",
    max_chars: int = 12000,
    overlap: int = 800,
) -> Dict[str, Dict[str, Any]]:
    qs = [q.strip() for q in questions if q and q.strip()]
    if not qs:
        return {}

    merged: Dict[str, Dict[str, Any]] = {
        q: {
            "value": None,
            "confidence": 0.0,
            "excerpt_location": {"page": None, "sentence": None},
        }
        for q in qs
    }

    chunks = _chunk_text(document_text, max_chars=max_chars, overlap=overlap)

    for idx, ch in enumerate(chunks, start=1):
        partial = answer_questions_json_evidence(
            model=model,
            document_text=ch,
            questions=qs,
            context_note=(
                context_note
                + f" This is chunk {idx}/{len(chunks)}. Answer only if the information is present in this chunk."
            ),
            temperature=0.1,
        )

        for q in qs:
            old_item = merged.get(q, {})
            new_item = partial.get(q, {})
            if _rich_answer_score(new_item) > _rich_answer_score(old_item):
                merged[q] = new_item

    return merged