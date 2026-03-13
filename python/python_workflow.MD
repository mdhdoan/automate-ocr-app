# Python / Streamlit Workflow Guide

## Overview

This application is a Python-first document processing workflow built in Streamlit for OCR, review, structured extraction, and export.

At a high level, the system works by combining:

1. **User-defined instructions**
2. **OCR-generated raw text**
3. **A long prompt for extraction**
4. **Human review and validation**

Everything the user configures inside the app becomes part of the final extraction context. That means the uploaded files, the project selection, the raw OCR text, the data standard, and the file structure guidance all contribute to one longer instruction chain that helps the extraction model understand **what the document is**, **how it is organized**, and **what fields need to be returned**.

---

## Core idea: everything feeds into a long prompt

The extraction step is not just “run OCR and get fields.”

Instead, the app builds a longer context for the language model by combining:

- the uploaded document content
- the OCR output
- the user’s **data standard**
- the user’s **file structure / layout notes**
- the extraction instructions for the target fields

This matters because many real-world forms are inconsistent, partially handwritten, scanned badly, or use shorthand. A raw OCR pass alone is often not enough to reliably extract structured data. The extra context helps the model interpret the text correctly.

### What the user provides
The user supplies two important instruction layers:

#### 1. Data Standard
The **data standard** explains what each target field means and how it should look.

A good data standard can include:
- field names
- definitions or descriptions
- expected syntax or format
- valid values or controlled vocabulary
- examples
- notes on ambiguity or exceptions

Example:
- `inspection_date`: Date the inspection took place, not the date the report was created
- Format: `YYYY-MM-DD`
- Example: `2004-10-22`

#### 2. File Structure / Layout Guidance
The **file structure** tells the model how the source document is usually organized.

This can include:
- where header information usually appears
- where comments or narrative notes are found
- where tables are located
- how repeated sections are structured
- how handwritten notes are typically mixed with printed text

This gives the model document-specific reading guidance instead of forcing it to guess from OCR text alone.

---

## End-to-end user workflow

## Step 1 — Select an existing project or create a new one

The workflow starts with the user choosing the project they want to work in.

A project acts as the working container for:
- uploaded files
- OCR results
- saved raw text
- data standards
- file structure guidance
- extraction outputs
- review state

### Why this matters
Projects keep different document batches, standards, and extraction setups separate from one another. This is especially important when different document types require different extraction logic.

### Typical user action
- Open the app
- Select an existing project from the list, or
- Create a new project for a new document type or dataset

---

## Step 2 — Upload files

Once the project is selected, the user uploads the source files.

Typical supported inputs include:
- PDF files
- image files
- text files

For scanned documents, PDFs are the most common starting point.

### What happens next
Uploaded files become the source material for OCR and downstream extraction.

---

## Step 3 — Define or edit the data standard

Before extraction, the user defines the target schema using a **data standard**.

This is one of the most important steps in the workflow because it tells the model exactly what information to return and how to interpret it.

### What should be included
For each field, the user should ideally provide:
- **Field name**
- **Definition / description**
- **Expected format**
- **Example value**
- **Optional notes or constraints**

### Why this matters
The model does better when it knows:
- what a field is supposed to represent
- how to distinguish similar fields
- what output formatting is expected
- what values are valid versus suspicious

Without a good data standard, extraction becomes much more guess-based.

---

## Step 4 — Define or edit the file structure guidance

Next, the user describes the document layout or file structure.

This gives the model extra reading instructions such as:
- where identifiers tend to appear
- where comments or observations are usually written
- which sections repeat
- how tables are arranged
- where handwritten notes are commonly located

### Why this matters
OCR gives text, but not always reliable interpretation.

The file structure layer helps the model understand the *shape* of the document, which improves extraction accuracy on semi-structured forms.

---

## Step 5 — OCR converts PDF pages into images

For PDF inputs, OCR begins by splitting the PDF into page images.

### Why the PDF is rendered to images
Many modern OCR and vision-language pipelines work best when each page is treated as an image. This makes it possible to:
- preserve visual layout
- process handwritten content
- handle mixed printed + handwritten forms
- maintain page-by-page review

### Result
A multi-page PDF becomes a sequence of page images.

Each page image is then passed to the OCR stage.

---

## Step 6 — OCR reads each page and generates raw text

Each page image is processed through the OCR pipeline to generate **raw text**.

The OCR step aims to:
- read printed text
- preserve section order where possible
- retain table-like structure as well as possible
- capture handwritten annotations when visible
- produce editable raw text for downstream cleanup

### Fast vs quality OCR
The workflow supports two OCR approaches or model modes:

#### Fast mode
Use this when:
- speed matters more than maximum detail
- documents are relatively clean
- the user wants quicker first-pass review

#### Quality mode
Use this when:
- documents are messy or low quality
- handwriting matters
- layout preservation matters more
- extraction accuracy is more important than speed

### Important note
OCR is not the final truth. It is the **working raw text layer** that the user can still review and correct.

---

## Step 7 — Merge page OCR into a single raw text result

After each page is processed, the application merges the page-level OCR outputs into one combined raw text representation for the document.

This merged raw text becomes the main editable source used by the extraction model.

### Why this matters
A single merged text layer makes it easier to:
- review the document as a whole
- edit OCR mistakes before extraction
- pass one consistent text block into the extraction prompt

---

## Step 8 — Review and edit the raw text

Once OCR is complete, the raw text is made available for user review.

This is a critical quality-control step.

### What the user does here
The user can:
- inspect OCR output
- correct obvious OCR mistakes
- fix broken lines or merged words
- clean up handwriting interpretation where needed
- confirm that the text is ready for extraction

### Why this matters
Even strong OCR models can make mistakes on:
- poor scans
- skewed pages
- tables
- handwriting
- unusual abbreviations

Allowing human edits before extraction significantly improves downstream field quality.

---

## Step 9 — Run LLM-based structured extraction

Once the raw text is ready, a second language model is used to extract structured information from the document.

This model uses:
- the merged raw text
- the user’s data standard
- the user’s file structure guidance
- the extraction instructions

### What the model is trying to do
The model reads the raw text and maps it into the target schema defined by the user.

In other words, it turns messy document text into structured fields.

### Example output types
- JSON-style records
- tabular field-value output
- normalized field sets ready for export

---

## Step 10 — Review results side by side

After extraction, the interface supports review by showing the source and processed outputs together.

The user can compare:
- the original PDF pages rendered as images
- the OCR raw text
- the extracted field table

### Why this layout matters
This gives the reviewer traceability across the full pipeline:
- image view shows what was actually on the page
- raw text view shows what OCR interpreted
- extraction table shows what the LLM concluded

This makes it much easier to spot whether an error came from:
- the original scan
- the OCR pass
- the extraction logic
- the data standard definition

---

## Step 11 — Human validation and sign-off

The final validation step is performed by a human.

This is intentional.

The app accelerates OCR and extraction, but the final decision on correctness should remain with the reviewer before export or operational use.

### Human reviewer responsibilities
The reviewer should confirm:
- the OCR text is acceptable
- extracted values match the source
- ambiguous fields were interpreted correctly
- required fields are complete
- output formatting is correct

### Why human review stays in the loop
Documents with handwritten notes, inconsistent layouts, or domain-specific language still benefit from human judgment. The tool is designed to reduce effort, not remove accountability.

---

## Step 12 — Export final results

After validation, the reviewed output can be exported for downstream use.

Typical export formats include:
- CSV
- JSON

### Use cases for export
- downstream analysis
- manual upload into another system
- record keeping
- QA audit trail
- batch ingestion into databases or other pipelines

---

## Practical pipeline summary

The full Python-side workflow can be summarized as:

1. Select or create a project
2. Upload files
3. Define or edit the data standard
4. Define or edit the file structure guidance
5. Split PDFs into page images
6. OCR each page
7. Merge page OCR into document-level raw text
8. Review and edit raw text
9. Run LLM extraction using the long prompt context
10. Compare source pages, raw text, and extracted results
11. Validate with a human reviewer
12. Export to CSV or JSON

---

## Why this design works

This workflow is designed to balance automation with reviewability.

### Benefits
- keeps document batches organized by project
- supports document-specific extraction rules
- improves extraction quality through data standards and file structure guidance
- allows OCR cleanup before extraction
- preserves human validation before export
- makes errors easier to trace and correct

### Tradeoff
The process is not a one-click black box. It intentionally keeps the user involved at the points where domain judgment matters most.
