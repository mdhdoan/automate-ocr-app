# app_streamlit_llm_extract.py
# Run: streamlit run app_streamlit_llm_extract.py

from __future__ import annotations

import re
import subprocess
from pathlib import Path
import streamlit as st

from llm_extract_core import (
    answer_questions_json,
    answer_questions_json_chunked,
)

DATA_ROOT = Path("/home/doanm/ocr-automate/data")

def parse_ctx_from_show(show_text: str) -> int | None:
    """
    Try to find a context size from `ollama show` output.
    This is heuristic; if not found, return None.
    """
    t = (show_text or "").lower()
    # look for common numbers printed in show output
    hits = [int(x) for x in re.findall(r"\b(2048|4096|8192|16384|32768|65536|131072)\b", t)]
    return max(hits) if hits else None

# ---------------- /data folder + file pickers ----------------
@st.cache_data(ttl=10)
def find_ocrtext_folders() -> list[Path]:
    """Find any subfolder under /data that ends with _ocrtext (any depth)."""
    if not DATA_ROOT.exists():
        return []
    folders = [p for p in DATA_ROOT.rglob("*") if p.is_dir() and p.name.endswith("_ocrtext")]
    folders.sort(key=lambda p: str(p).lower())
    return folders


@st.cache_data(ttl=10)
def list_theme_txt_files(folder: Path) -> list[Path]:
    """List only files that end with 'theme.txt' (case-insensitive) inside the selected folder (non-recursive)."""
    if not folder.exists() or not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.is_file()]
    files.sort(key=lambda p: p.name.lower())
    return files


# ---------------- Token estimation ----------------
def estimate_tokens(text: str) -> int:
    """
    Best-effort token estimate:
    1) If tiktoken exists, use it (roughly GPT tokenization).
    2) Otherwise fallback: ~4 chars per token.
    """
    text = text or ""
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, int(len(text) / 4))


# ---------------- Ollama model list + show ----------------
@st.cache_data(ttl=10)
def list_ollama_models() -> list[str]:
    """List locally installed Ollama models via CLI."""
    try:
        out = subprocess.check_output(["ollama", "list"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return []

    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not lines:
        return []
    if "NAME" in lines[0].upper():
        lines = lines[1:]

    models = []
    for ln in lines:
        name = ln.split()[0]
        if name:
            models.append(name)

    # de-dup preserve order
    seen, uniq = set(), []
    for m in models:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq


@st.cache_data(ttl=60)
def ollama_show_raw(model: str) -> str:
    """Return raw output of `ollama show <model>`."""
    try:
        out = subprocess.check_output(["ollama", "show", model], text=True, stderr=subprocess.STDOUT)
        return out.strip()
    except subprocess.CalledProcessError as e:
        return (e.output or "").strip() or f"Failed to run: ollama show {model}"
    except Exception as e:
        return f"Failed to run: ollama show {model}\n{e}"


def infer_tags_from_show(model: str, show_text: str) -> list[str]:
    """Heuristic tags (no manual tagging required)."""
    n = (model or "").lower()
    t = (show_text or "").lower()
    tags: list[str] = []

    if "vision" in n or "vision" in t or "vl" in n:
        tags.append("vision")
    if "embed" in n or "embedding" in t:
        tags.append("embeddings")
    if "code" in n or "coder" in n or "code" in t:
        tags.append("code")
    if "instruct" in n or "instruct" in t:
        tags.append("instruct")
    if "json" in t or "structured" in t or "function" in t:
        tags.append("structured/json")

    m = re.search(r"[:\-]([0-9]+)b\b", n)
    if m:
        b = int(m.group(1))
        tags.append("fast" if b <= 4 else ("balanced" if b <= 9 else "quality"))

    ctx_nums = [int(x) for x in re.findall(r"\b(8192|16384|32768|65536|131072)\b", t)]
    if ctx_nums and max(ctx_nums) >= 32768:
        tags.append("long-doc")

    if not tags:
        tags.append("general")

    # de-dup preserve order
    seen = set()
    out = []
    for x in tags:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


@st.cache_data(ttl=60)
def model_meta(model: str) -> dict:
    show_text = ollama_show_raw(model)
    tags = infer_tags_from_show(model, show_text)
    return {"show": show_text, "tags": tags}


# ---------------- UI ----------------
st.set_page_config(page_title="LLM Q&A → JSON Extractor", layout="wide")
st.title("LLM Q&A → JSON Extractor")

left, right = st.columns([1, 1.2], gap="large")

with left:
    st.subheader("Inputs")

    # Model picker
    models = list_ollama_models()

    c1, c2 = st.columns([1, 1])
    with c1:
        annotate = st.toggle("Annotate dropdown with tags", value=True)
    with c2:
        if st.button("Refresh models"):
            st.cache_data.clear()
            models = list_ollama_models()

    if not models:
        st.warning("No Ollama models detected. Make sure `ollama` is installed and available in PATH.")
        model = st.text_input("Ollama model (manual)", value="gemma3")
        meta = {"tags": ["unknown"], "show": ""}
    else:
        default_model = "gemma3" if "gemma3" in models else models[0]

        if annotate:
            def _fmt(m: str) -> str:
                mm = model_meta(m)
                return f"{m}  —  {', '.join(mm['tags'])}"
            model = st.selectbox("Ollama model", models, index=models.index(default_model), format_func=_fmt)
        else:
            model = st.selectbox("Ollama model", models, index=models.index(default_model))

        meta = model_meta(model)
        st.caption(f"Selected tags: **{', '.join(meta['tags'])}**")

        with st.expander("Show model details (ollama show)"):
            st.code(meta["show"], language="text")

    context_note = st.text_area(
        "Context note (optional)",
        value="This is a legal document between 2 entities. The information might be spread out through the entire document",
        height=90,
    )

    questions_text = st.text_area(
        "Questions (one per line)",
        value=(
            "What is the agreement ID\n"
            "Who is the lead organization\n"
            "What's the date of the agreement\n"
            "What is the project duration\n"
            "What's the title of the agreement\n"
            "What's the total money involved"
        ),
        height=180,
    )

    st.markdown("---")
    st.subheader("Document input")

    uploaded = st.file_uploader("Upload a .txt file", type=["txt"])

    # Folder dropdown: any /data/**/_ocrtext
    folders = find_ocrtext_folders()
    folder_labels: list[str] = []
    folder_map: dict[str, Path] = {}

    if folders:
        for p in folders:
            try:
                rel = p.relative_to(DATA_ROOT)
                label = f"/data/{rel.as_posix()}"
            except Exception:
                label = str(p)
            folder_labels.append(label)
            folder_map[label] = p
    else:
        st.caption("No /data/*_ocrtext folders found (or /data not accessible).")

    selected_folder_label = st.selectbox(
        "…or select a /data folder ending with _ocrtext",
        options=["(none)"] + folder_labels,
        index=0,
    )

    selected_file_path: Path | None = None
    if selected_folder_label != "(none)":
        chosen_folder = folder_map[selected_folder_label]
        theme_files = list_theme_txt_files(chosen_folder)

        if not theme_files:
            st.warning("No files ending with 'theme.txt' found in that folder.")
        else:
            file_names = [p.name for p in theme_files]
            chosen_name = st.selectbox("Select a theme.txt file", options=file_names, index=0)
            selected_file_path = next(p for p in theme_files if p.name == chosen_name)

    pasted = st.text_area("…or paste document text", height=220)

    st.markdown("---")
    st.subheader("Mode")

    force_chunking = st.toggle("Force chunking mode", value=False)
    TOKEN_THRESHOLD = st.number_input(
        "Auto-switch to chunking above (estimated tokens)",
        min_value=5_000,
        max_value=120_000,
        value=25_000,  # dense OCR legal ~30 pages often 20k–35k tokens
        step=5_000,
        help="OCR legal docs inflate tokens; 25k is a good ~30-page switch point.",
    )

    run = st.button("Extract", type="primary")

    # Determine the input text ONCE (avoid double-reading uploads)
    doc_text = ""
    origin_label = ""

    if selected_file_path is not None:
        doc_text = selected_file_path.read_text(encoding="utf-8", errors="replace")
        origin_label = str(selected_file_path)
    elif uploaded is not None:
        doc_text = uploaded.getvalue().decode("utf-8", errors="replace")
        origin_label = uploaded.name
    else:
        doc_text = pasted
        origin_label = "pasted"

with right:
    st.subheader("Result")

    if run:
        questions = [q.strip() for q in questions_text.splitlines() if q.strip()]

        if not doc_text.strip():
            st.error("Provide a .txt file, select a theme.txt from /data, or paste text.")
        elif not questions:
            st.error("Add at least one question.")
        else:
            tok_est = estimate_tokens(doc_text)
            ctx = parse_ctx_from_show(meta.get("show",""))
            # If we can detect context, chunk when we exceed ~80% of it
            ctx_threshold = int(ctx * 0.8) if ctx else None

            # Effective threshold is the smaller of:
            # - your user threshold (25k)
            # - model context threshold (e.g. 8192 * 0.8 = 6553)
            effective_threshold = int(TOKEN_THRESHOLD)
            if ctx_threshold:
                effective_threshold = min(effective_threshold, ctx_threshold)

            auto_chunk = tok_est > effective_threshold
            use_chunked = force_chunking or auto_chunk

            st.caption(
                f"Estimated tokens: **{tok_est:,}** | "
                f"Effective chunk threshold: **{effective_threshold:,}** | "
                f"Mode: **{'Chunk + merge' if use_chunked else 'Single-pass'}**"
            )

            st.write("Input preview:", doc_text[:800])

            chunk_chars = 12000
            overlap_chars = 800
            if use_chunked:
                with st.expander("Chunking settings"):
                    chunk_chars = st.slider("Chunk size (chars)", 4000, 30000, 12000, step=1000)
                    overlap_chars = st.slider("Overlap (chars)", 0, 5000, 800, step=100)

            with st.spinner("Asking the model…"):
                if use_chunked:
                    result = answer_questions_json_chunked(
                        model=model,
                        document_text=doc_text,
                        questions=questions,
                        context_note=context_note,
                        max_chars=int(chunk_chars),
                        overlap=int(overlap_chars),
                    )
                else:
                    result = answer_questions_json(
                        model=model,
                        document_text=doc_text,
                        questions=questions,
                        context_note=context_note,
                    )

            st.json(result)
    else:
        st.info("Select a theme.txt / upload / paste, add questions, then click Extract.")
