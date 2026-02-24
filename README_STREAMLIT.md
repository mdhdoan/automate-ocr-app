# automate-ocr-app — Python / Streamlit

Local-first document ingestion + OCR + structured field extraction using **Ollama**.

This README covers the **Python / Streamlit** app. For the **R / Shiny** app, see `README_SHINY.md` (and `R/README.md` for the most detailed R notes).

---

## What it does

- Upload PDF / image / txt files into a **Project**
- Build a **text layer**
  - extract embedded text from text-native PDFs
  - OCR scanned PDFs/images using an Ollama **vision** model (PDF pages rendered to PNG first)
- Define a versioned **Data Standard** (fields, definitions, required, regex/format, allowed values)
- Run extraction via an Ollama **text** model into JSON outputs
- Review/validate outputs with rule-based flags (missing required fields, format errors, etc.)

---

## Repo layout (relevant parts)

- `python/app_streamlit_llm_extract.py` — Streamlit entrypoint
- `python/ocr.py` — CLI OCR for folders of images
- `python/convert_to_img.py` — CLI PDF → PNG
- `prompts/` — prompt assets / experiments (WIP)

---

## Quick start

### 1) Prerequisites

- Python 3.10+ (3.11 recommended)
- **Ollama** installed and running
- One **vision-capable** model (OCR) + one **text** model (extraction)

Pull models (examples — use the names you standardize on):
```bash
ollama pull <vision-model>
ollama pull <text-model>
```

### 2) Install dependencies (choose one)

**Option A — install from the repo snapshot (slow/heavy but reproducible):**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Option B — minimal install for Streamlit (fast):**
```bash
python -m venv .venv
source .venv/bin/activate
pip install streamlit pandas ollama PyMuPDF
```

> Notes:
> - `PyMuPDF` is used to read/render PDFs.
> - If your environment differs (Ubuntu vs macOS), you may need OS-level deps for PDF/image support.

### 3) Configure where projects are saved

The Streamlit app currently writes project data to a hard-coded location:
`/home/doanm/ocr-automate/data`

Pick one:

**Option A (no code change):**
```bash
sudo mkdir -p /home/doanm/ocr-automate/data
sudo chown -R $USER:$USER /home/doanm/ocr-automate
```

**Option B (recommended): change `DATA_ROOT`**
In `python/app_streamlit_llm_extract.py`, change:
```py
DATA_ROOT = Path("/home/doanm/ocr-automate/data")
```
to:
```py
DATA_ROOT = Path("./data")
```

### 4) Run the app

From repo root:
```bash
streamlit run python/app_streamlit_llm_extract.py
```

---

## How the Streamlit workflow is organized

### Projects & versions
Each “Project” is stored on disk with versioned artifacts:
- **Files**: originals you uploaded
- **Text**: extracted/OCR’d text (`.txt`)
- **Previews**: rendered page images (`.png`) for review
- **Standards**: versioned JSON `standard_v*.json`
- **Structure notes**: versioned `structure_v*.txt`
- **Runs**: extraction outputs `run_*.json`
- **Exports**: CSV/JSON exports
- **Audit log**: append-only `audit.jsonl`

### Review & validation
The review experience supports a “validator-style” workflow:
- focus on missing required fields
- format/regex checks
- quickly iterate on standards/structure notes and re-run extraction

---

## CLI utilities

### OCR a folder of images
```bash
python python/ocr.py <vision-model> <image_dir> --recursive
```

Optional custom prompt:
```bash
python python/ocr.py <vision-model> <image_dir> --prompt "Extract all readable text. Return plain text only."
```

### Convert PDFs to PNGs
```bash
python python/convert_to_img.py <pdf_dir>
python python/convert_to_img.py <pdf_dir> --recursive
```

---

## Troubleshooting

### OCR returns empty/garbled text
- Confirm Ollama is running (default: `http://localhost:11434`)
- Ensure you chose a **vision** model for OCR (text-only models won’t work)
- For PDFs: large DPI/zoom can be slow; reduce render scale or limit pages

### JSON output is malformed
- Use a stronger text model for extraction
- Reduce the number of fields per run
- Tighten your Data Standard (clear definitions + examples + explicit formats)

### Preview images don’t generate
- Check `PyMuPDF` install
- Ensure your project folder is writable
- Confirm PDFs aren’t encrypted/password protected

---

## Suggested next improvements (optional)

- Make `DATA_ROOT` configurable via environment variable
- Add per-field citations (page number + snippet) for extracted values
- Add a batch queue + progress persistence for bulk processing
- Add schema validation using Pydantic / JSON Schema

---

## Contributing

PRs welcome. If you’re making larger changes:
- keep the default workflow local-first (no uploads to external services by default)
- preserve versioned outputs (standards/structures/runs)
