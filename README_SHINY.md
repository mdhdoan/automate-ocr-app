# automate-ocr-app — R / Shiny

Local-first OCR + extraction using **Ollama**, with a Shiny UI for single-document and bulk workflows.

This README covers the **R / Shiny** app. For the Python / Streamlit app, see `README_STREAMLIT.md`.

> The most detailed setup notes for the R app may already be in `R/README.md`.

---

## What it does

- Load PDFs/images and run OCR through an Ollama **vision** model
- Extract structured fields / summaries through an Ollama **text** model
- Support “profiles” (prompt/field presets) for repeated workflows
- Bulk mode for processing multiple files with consistent settings

---

## Repo layout (relevant parts)

- `R/app.R` — Shiny entrypoint
- `R/README.md` — detailed R-specific install/run notes (recommended)

---

## Quick start (typical)

### 1) Prerequisites

- R 4.2+ (or your org standard)
- RStudio (optional but convenient)
- **Ollama** installed and running
- One **vision-capable** model (OCR) + one **text** model (extraction)

Pull models (examples — use the names you standardize on):
```bash
ollama pull <vision-model>
ollama pull <text-model>
```

### 2) Install R packages

Follow `R/README.md` for the canonical list.

If you want a simple start, install what `R/app.R` imports (you can open it to confirm), and then add missing packages as prompted by Shiny.

### 3) Run

From within the `R/` folder:
```r
shiny::runApp()
```

Or from repo root:
```r
shiny::runApp("R")
```

---

## Configuration

Common things you may need to set (depending on how the app is wired):

- **Ollama host** (default is usually `http://localhost:11434`)
- **Model names** used for OCR vs extraction
- **Output directory** for bulk runs (so results are easy to review/export)

Check `R/README.md` (and/or constants at the top of `R/app.R`) for the exact variables/fields.

---

## Troubleshooting

### Shiny opens but extraction fails
- Confirm Ollama is running locally
- Ensure model names exactly match what you pulled in Ollama
- Try a smaller file first (one page PDF, one image)

### OCR is slow on PDFs
- PDFs are often converted page-by-page; large page counts will take time
- Reduce render scale/DPI if the app exposes it
- Process only selected pages (if supported) to validate the workflow first

### Package install issues
- On Ubuntu, you may need system libraries for curl/ssl/xml and image handling
- Use your org’s recommended R setup method (renv, system libs, etc.)

---

## Recommended next improvements (optional)

- Add a small “health check” panel (Ollama reachable, models present)
- Add run manifests (inputs, models, prompts, timestamps) for reproducibility
- Add export formats (CSV + JSON) with consistent naming conventions

---

## See also

- `README_STREAMLIT.md` — Python / Streamlit workflow
- `R/README.md` — deeper R setup notes
