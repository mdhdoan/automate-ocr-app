# Local OCR + Field Extraction App (Shiny + Ollama)

This repository contains a local-only R Shiny application that:

1. Performs OCR on PDF / PNG / JPG files using an Ollama vision-capable model (default: mistral-small3.2).
2. Sends the OCR text to an Ollama text model (default: gpt-oss) to extract named fields with confidence scores.
3. Supports both Single-document and Bulk-document workflows.
4. Allows per-page OCR review and editing before extraction.
5. Supports saving and reusing “profiles” for different document types.
6. Exports results to CSV, JSON, and OCR text.

Everything runs locally: Shiny, Ollama, and models are on the same machine.

---

## 1. Requirements

### 1.1 System

* R (4.x recommended)
* A machine capable of running Ollama and your chosen models (CPU works; GPU helps)
* Linux, macOS, or Windows with Ollama installed

### 1.2 R packages

Required packages:

* shiny
* bslib
* ollamar
* jsonlite
* pdftools

Package roles:

* shiny: web UI + reactive server
* bslib: theme support and dark mode switching
* ollamar: calls to Ollama models for OCR and extraction
* jsonlite: profile storage and JSON parsing/writing
* pdftools: PDF page rendering and page counting

### 1.3 Ollama and models

You need Ollama installed and running, plus:

* A vision-capable model for OCR (default: mistral-small3.2)
* A text model for extraction (default: gpt-oss)

By default, the app assumes Ollama is reachable at [http://localhost:11434](http://localhost:11434). If your host or port differs, configure it in your app startup (for example by setting the ollamar default URL near the top of app.R).

---

## 2. Project structure

Expected layout:

* app.R
* profiles directory (auto-created on first run)

Profiles are saved as JSON files inside the profiles directory.

---

## 3. How the app works

### 3.1 OCR stage

Input: a PDF, PNG, JPG, or JPEG.

Process:

* If the file is a PDF:

  * The PDF is converted into page images using pdftools at a configurable DPI (default 200).
  * Optionally, only a page range is processed (First page and Last page).
* If the file is an image:

  * It is treated as a single-page document.

For each page image:

* The app calls the OCR model through Ollama and requests plain text output in reading order.
* The output is stored per page and also combined into a single document text with page separators.

OCR output forms:

* Text per page (used for per-page review/editing)
* Combined OCR text (used for extraction and exports)

### 3.2 Per-page OCR editing

Both Single and Bulk modes allow you to:

* Select a page
* Edit the OCR text for that page
* Save the edit (stored separately from the raw OCR output)
* Reset the page to the original OCR output

When extraction or OCR-text export is run, the edited per-page text is used whenever edits exist.

### 3.3 Field extraction stage

Inputs:

* A list of fields (one per line)
* Optional extra instructions (user prompt)
* OCR text (either from the OCR stage or an uploaded OCR text file)

Process:

* The app constructs a strict system instruction requiring JSON-only output with the structure:

  * fields: an array of objects with name, value, and confidence in the range 0–1
  * Missing or non-inferable fields must return an empty value and confidence 0
* The model is called using:

  * Completion-style calls for gpt-oss
  * Chat-style calls for other model names
* The response is parsed into a normalized table:

  * name
  * value
  * confidence
* The raw model response is also stored for debugging.

Output views:

* Extracted Fields table (primary)
* Raw JSON text (debug view)

---

## 4. Running the app

1. Ensure Ollama is installed and running.
2. Ensure your OCR and extraction models are available in Ollama.
3. Place app.R in a folder.
4. Launch the Shiny app from R (for example using shiny’s standard run method).
5. Open the local URL printed by R (or the browser window that opens automatically).

---

## 5. Using the app

### 5.1 Single mode workflow

Step 1: OCR

* Upload a file under Single file.
* If the file is a PDF, adjust DPI and optionally set First page and Last page.
* Set the OCR model name.
* Click Run OCR.
* Review the page image and OCR text side by side.
* Edit the OCR text per page as needed, then Save edit or Reset.

Step 2: Extraction

* Enter the field list (one field per line).
* Optionally update the extra instructions to guide extraction.
* Set the extraction model name.
* Optionally upload a text file to use as OCR input instead of running OCR.
* Click Run Extraction.
* View results in Extracted Fields and Raw JSON.

Step 3: Export

* CSV export: table of name, value, confidence
* JSON export: keyed by field name with value and confidence
* OCR text export: combined OCR text with page separators (edits preferred)

### 5.2 Bulk mode workflow

Step 1: Bulk OCR

* Upload multiple documents under Bulk docs.
* Click 1) Bulk OCR.
* Select a file and page to review.
* Edit and save/reset OCR text per page as needed.

Step 2: Bulk Extract

* Enter the field list and extra instructions.
* Click 2) Bulk Extract.
* The app extracts fields for each file independently.
* The app aggregates results into a single table:

  * Rows: fields
  * Columns: filenames
* Download Bulk CSV or Bulk JSON.

Bulk exports:

* Bulk CSV: aggregated table (fields x files)
* Bulk JSON: keyed by field name, with each field containing values per file

---

## 6. Profiles (save / load / delete)

Profiles let you reuse configurations for different document types.

### 6.1 What a profile stores

* field_list
* user_prompt
* ocr_model
* extract_model
* dpi

Profiles are saved as JSON files in the profiles directory next to app.R.

### 6.2 Saving a profile

* Configure field list, prompt, models, and DPI
* Enter a profile name
* Click Save
* The name is sanitized for safe filenames (special characters become underscores)
* The profile appears in the Load profile dropdown

### 6.3 Loading a profile

* Choose a profile from the dropdown
* Click Load
* The UI updates to the saved values

### 6.4 Deleting a profile

* Select a profile
* Click Delete
* The profile JSON file is removed permanently

---

## 7. Data flow manual

### Single mode data flow

1. User uploads a file
2. OCR stage produces:

   * page images
   * OCR text per page
   * combined OCR text
3. Optional edits update the per-page OCR text
4. Extraction uses:

   * combined OCR text built from edited pages (if edits exist)
5. Exports use:

   * extracted fields table (CSV/JSON)
   * combined OCR text (OCR export)

### Bulk mode data flow

1. User uploads multiple files
2. Bulk OCR produces, per file:

   * page images
   * OCR text per page
3. Optional edits update per-file per-page text
4. Bulk extraction runs per file using combined edited text
5. Aggregation merges outputs into:

   * a single table of fields x filenames
6. Bulk exports are generated from the aggregated table

---

## 8. Troubleshooting

### OCR fails

Possible causes:

* Ollama is not running or not reachable on the configured host/port
* OCR model is not vision-capable
* The model name is misspelled or not pulled
* PDF rendering issues at high DPI (memory or time)

Things to try:

* Confirm Ollama is running locally
* Try a different OCR vision model
* Lower DPI for large PDFs
* Try a smaller PDF page range

### Extraction returns empty response or JSON parse errors

Possible causes:

* The extraction model is weak for strict JSON output
* OCR text is too long or noisy
* The model returns extra prose around JSON

Things to try:

* Use a stronger extraction model
* Reduce the number of fields
* Improve OCR quality (DPI, better scan)
* Use the Raw JSON tab to see what was returned

---

## 9. Known limitations

* OCR quality depends heavily on the selected vision model and scan quality.
* Complex tables, checkboxes, and handwriting may require better prompts, stronger models, or multi-step extraction.
* OCR text is truncated before extraction to reduce context overflow; extremely long documents may lose late-page information during extraction.
* JSON parsing can fail if the model output is malformed.

---

## 10. Customization ideas

* Add preset profiles for common form types (for example SIL, BC16).
* Add optional post-processing:

  * numeric validation
  * date normalization
  * controlled vocab checks
* Add logging, QA/QC flags, or per-field verification UI.
* Add per-page extraction mode (instead of whole-document extraction).

---

## 11. Quick start summary

* Install required R packages
* Install and run Ollama
* Ensure a vision OCR model and an extraction model are available
* Launch the Shiny app locally
* Run OCR, optionally edit per-page text, run extraction, export results
* Save profiles for repeat workflows
