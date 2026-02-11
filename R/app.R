# app.R --------------------------------------------------------------
# Local Shiny app:
# 1) OCR PDF/PNG/JPEG via Ollama + mistral-small3.2
# 2) Extract fields with confidence via Ollama + gpt-oss
# 3) User can customize field list + user prompt
# 4) Save / load / delete profiles (field list, user prompt, models, dpi)
# 5) Export results as CSV or JSON

# install.packages(c("shiny", "magick", "ollamar", "jsonlite"))  # if needed

library(shiny)
library(bslib)
library(magick)
library(ollamar)
library(jsonlite)
library(pdftools)


## If Ollama is not on localhost:11434, point to your URL:
# set_default_url("http://localhost:11434")

options(shiny.maxRequestSize = 50 * 1024^2)  # 50 MB

# ---- Profiles config -----------------------------------------------

profiles_dir <- "profiles"
if (!dir.exists(profiles_dir)) dir.create(profiles_dir, recursive = TRUE)

list_profiles <- function() {
  files <- list.files(profiles_dir, pattern = "\\.json$", full.names = FALSE)
  sub("\\.json$", "", files)  # drop .json extension
}

sanitize_profile_name <- function(x) {
  x <- trimws(x)
  x <- gsub("[^A-Za-z0-9_-]", "_", x)  # letters, digits, underscore, hyphen
  x
}

save_profile_to_disk <- function(name, input) {
  prof_name <- sanitize_profile_name(name)
  if (prof_name == "") stop("Profile name is empty after sanitization.")

  profile <- list(
    field_list    = input$field_list,
    user_prompt   = input$user_prompt,
    ocr_model     = input$ocr_model,
    extract_model = input$extract_model,
    dpi           = input$dpi
  )

  path <- file.path(profiles_dir, paste0(prof_name, ".json"))
  jsonlite::write_json(profile, path, pretty = TRUE, auto_unbox = TRUE)
  prof_name
}

load_profile_from_disk <- function(name) {
  prof_name <- sanitize_profile_name(name)
  path <- file.path(profiles_dir, paste0(prof_name, ".json"))
  if (!file.exists(path)) stop("Profile not found: ", prof_name)
  jsonlite::fromJSON(path, simplifyVector = TRUE)
}

delete_profile_from_disk <- function(name) {
  prof_name <- sanitize_profile_name(name)
  path <- file.path(profiles_dir, paste0(prof_name, ".json"))
  if (!file.exists(path)) stop("Profile not found: ", prof_name)
  file.remove(path)
}

# ---- Example docs --------------------------------------------------

example_dir <- "example_docs"
if (!dir.exists(example_dir)) dir.create(example_dir, recursive = TRUE)

# Put your sample PDF(s) in ./example_docs/
# Example: ./example_docs/BC16-056220_Area_3_Ansedagan_Creek_2004_Format_6C.pdf
example_docs <- list.files(
  example_dir,
  pattern = "\\.(pdf|png|jpg|jpeg)$",
  full.names = TRUE,
  ignore.case = TRUE
)

# Named choices for UI
example_choices <- setNames(example_docs, basename(example_docs))

# Example OCR text files for extraction (put .txt files in ./example_docs/)
example_txts <- list.files(
  example_dir,
  pattern = "\\.(txt|text|log)$",
  full.names = TRUE,
  ignore.case = TRUE
)
example_txt_choices <- setNames(example_txts, basename(example_txts))


# ---- OCR helpers ---------------------------------------------------

pdf_to_png_pages <- function(pdf_path, dpi = 200L) {
  if (!file.exists(pdf_path)) {
    stop("PDF not found: ", pdf_path)
  }

  # pdftools converts each page to a PNG and returns their paths
  png_paths <- pdftools::pdf_convert(
    pdf = pdf_path,
    dpi = dpi
  )

  if (length(png_paths) == 0L) {
    stop("No pages converted from PDF: ", pdf_path)
  }

  png_paths
}


ocr_page_image_ollama <- function(
  image_path,
  model  = "mistral-small3.2",
  prompt = "Transcribe all text in this image as plain UTF-8 text, preserving reading order.",
  temperature = 0, 
  keep_alive
) {
  if (!file.exists(image_path)) {
    stop("Image file not found: ", image_path)
  }

  generate(
    model       = model,
    prompt      = prompt,
    images      = image_path,  # local path, ollamar sends it to Ollama
    stream      = FALSE,
    output      = "text",
    temperature = temperature,
    keep_alive = '5m'
  )
}

ocr_any_file_ollama <- function(
  file_path,
  model  = "mistral-small3.2",
  dpi    = 200L,
  prompt = "Transcribe all text in this scanned page as plain UTF-8 text, preserving reading order.",
  progress_callback = NULL,
  pages = NULL  # <--- NEW
) {
  ext <- tolower(tools::file_ext(file_path))

  if (ext == "pdf") {
    png_paths <- pdf_to_png_pages(file_path, dpi = dpi)

    # If a page subset is requested, filter paths accordingly
    if (!is.null(pages)) {
      n_total <- length(png_paths)
      pages <- unique(as.integer(pages))
      pages <- pages[pages >= 1 & pages <= n_total]

      if (length(pages) == 0L) {
        stop("No valid pages to OCR after applying page range.")
      }

      png_paths <- png_paths[pages]
      page_numbers <- pages
    } else {
      page_numbers <- seq_along(png_paths)
    }

  } else if (ext %in% c("png", "jpg", "jpeg")) {
    png_paths    <- file_path
    page_numbers <- 1L
  } else {
    stop("Unsupported file type: ", ext)
  }

  n_pages <- length(png_paths)
  page_texts <- character(n_pages)

  for (i in seq_len(n_pages)) {
    if (!is.null(progress_callback)) {
      progress_callback(i, n_pages)
    }
    page_texts[i] <- ocr_page_image_ollama(
      image_path  = png_paths[i],
      model       = model,
      prompt      = prompt,
      temperature = 0
    )
  }

  combined <- paste(
    sprintf("---- PAGE %d ----\n%s", page_numbers, page_texts),
    collapse = "\n\n"
  )

  list(
    text_per_page = page_texts,
    combined_text = combined,
    page_paths    = png_paths,
    page_numbers  = page_numbers  # <--- NEW
  )
}

# ---- JSON helper for extraction -----------------------------------

extract_json_substring <- function(x) {
  # take from first "{" to last "}"
  start <- regexpr("\\{", x)
  end   <- regexpr(".*\\}", x)
  if (start[1] == -1 || end[1] == -1) return(x)
  substr(x, start[1], start[1] + attr(end, "match.length") - 1)
}

# ---- Shiny UI ------------------------------------------------------

light_theme <- bs_theme(
  version   = 5,
  bootswatch = "lumen"
)

dark_theme <- bs_theme(
  version   = 5,
  bootswatch = "cyborg"
)

ui <- fluidPage(
  theme = dark_theme,
  tags$head(
    tags$style(HTML("
      .ocr-spinner-container {
        margin-top: 10px;
        min-height: 32px;
      }

      /* Make sure preview images never overflow their columns */
      #original_page_image img,
      #bulk_page_image img {
        max-width: 100%;
        height: auto;
        display: block;
      }

      /* Scrollable panel for OCR text (single + bulk) */
      .ocr-page-text-wrapper {
        max-height: 80vh;
        overflow-y: auto;
        padding: 0.5rem;
        border-radius: 0.5rem;
      }

      /* Wrap long lines in verbatim / text outputs */
      #ocr_output,
      #raw_json_output,
      #bulk_page_text {
        white-space: pre-wrap;
        word-wrap: break-word;
      }

      textarea#edit_page_text,
      textarea#bulk_edit_page_text {
        width: 100% !important;
        max-width: 100%;
        box-sizing: border-box;
        font-family: monospace;

        white-space: pre-wrap;   /* ✅ preserves line breaks AND wraps */
        word-wrap: break-word;  /* ✅ breaks long unspaced words */
        overflow-wrap: break-word;

        line-height: 1.4;
      }

      
      /* Light vs dark subtle background for the OCR panel */
      @media (prefers-color-scheme: dark) {
        .ocr-page-text-wrapper {
          border: 1px solid rgba(255,255,255,0.15);
          background-color: rgba(255,255,255,0.03);
        }
      }

      @media (prefers-color-scheme: light) {
        .ocr-page-text-wrapper {
          border: 1px solid rgba(0,0,0,0.1);
          background-color: rgba(0,0,0,0.02);
        }
      }
            .page-diff-panel {
        max-height: 40vh;
        overflow-y: auto;
        margin-top: 0.5rem;
        padding: 0.5rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(255,255,255,0.15);
        font-family: monospace;
        font-size: 0.9rem;
        white-space: pre-wrap;
      }

      .diff-line-same {
        opacity: 0.6;
      }

      .diff-line-added {
        background-color: rgba(0, 255, 0, 0.1);
      }

      .diff-line-removed {
        background-color: rgba(255, 0, 0, 0.08);
        text-decoration: line-through;
      }

      .diff-line-modified {
        background-color: rgba(255, 215, 0, 0.12);
      }

    ")),
    # Detect system dark-mode preference and send to Shiny
    tags$script(HTML("
      $(function() {
        if (window.matchMedia) {
          var mq = window.matchMedia('(prefers-color-scheme: dark)');
          Shiny.setInputValue('system_pref_dark', mq.matches, {priority: 'event'});
        }
      });
    "))
  ),

  titlePanel("Doc Flow: Local OCR & Details Extraction (with Mistral and GPT-OSS)"),
  
  sidebarLayout(
    sidebarPanel(
      h4("Appearance"),
      checkboxInput(
        "dark_mode",
        "Dark mode",
        value = FALSE
      ),     
      tags$hr(),
      h4("1. Choose document"),
      radioButtons(
        "doc_source",
        label = NULL,
        choices = c("Example" = "example", "Upload" = "upload"),
        selected = "upload",
        inline = TRUE
      ),

      conditionalPanel(
        condition = "input.doc_source == 'upload'",
        fileInput(
          "file",
          "PDF / PNG / JPEG",
          accept = c(".pdf", ".png", ".jpg", ".jpeg")
        )
      ),

      conditionalPanel(
        condition = "input.doc_source == 'example'",
        selectInput(
          "example_doc",
          "Example document",
          choices  = example_choices,
          selected = if (length(example_choices)) unname(example_choices[1]) else NULL
        ),
        actionButton("use_example", "Use this example", class = "btn-secondary")
      ),
      uiOutput("selected_doc_info"),
      sliderInput(
        "dpi",
        "PDF render DPI (for OCR) - higher = better quality, but slower",
        min = 100,
        max = 300,
        value = 100,
        step = 50
      ),
      numericInput("page_start", "First page to OCR (blank = 1)", value = NA, min = 1, step = 1),
      numericInput("page_end",   "Last page to OCR (blank = all pages) - limit pages for speed", value = NA, min = 1, step = 1),
      textInput(
        "ocr_model",
        "OCR model (default: mistral-small3.2, run on-prem with Ollama)",
        value = "mistral-small3.2"
      ),

      actionButton("run_ocr", "Run OCR", class = "btn-primary"),
      uiOutput("ocr_spinner"),
      h5("1.1 Optional: Export OCR text"),
      downloadButton("download_ocr_text", "Download OCR text (.txt)"),    
      br(),
      tags$hr(),

      h4("2. Field extraction (default to gpt-oss)"),
      fileInput(
        "extract_text_file",
        "Optional: upload OCR text (.txt) to use for extraction",
        accept = c(".txt", ".text", ".log")
      ),
      selectInput(
        "example_extract_text",
        "Or pick example OCR text (from example_docs)",
        choices  = example_txt_choices,
        selected = if (length(example_txt_choices)) unname(example_txt_choices[1]) else NULL
      ),
      actionButton("use_example_extract_text", "Use this example text", class = "btn-secondary"),
      br(),
      tags$hr(),
      textAreaInput(
        "field_list",
        "Fields to extract (one per line)",
        value = "Field 1\nField 2\nField 3",
        rows = 5
      ),
      textAreaInput(
        "user_prompt",
        "User prompt (additional instructions for extraction model)",
        value = "Extract the requested fields as accurately as possible from the OCR text.",
        rows = 4
      ),
      textInput(
        "extract_model",
        "Extraction model (default: GPT-OSS, run on-prem with Ollama)",
        value = "gpt-oss"
      ),
    
      actionButton("run_extract", "Run Field Extraction", class = "btn-success"),      
      tags$hr(),

      h4("Profiles"),
      textInput(
        "profile_name",
        "Profile name (for save)",
        value = ""
      ),
      fluidRow(
        column(6, actionButton("save_profile", "Save Profile")),
        column(6, actionButton("delete_profile", "Delete Profile", class = "btn-danger"))
      ),
      br(),
      selectInput(
        "load_profile_name",
        "Load existing profile",
        choices = list_profiles(),
        selected = NULL
      ),
      actionButton("load_profile", "Load Profile"),
      tags$hr(),

      h4("3. Export Extracted date"),
      downloadButton("download_csv", "Download as CSV"),
      downloadButton("download_json", "Download as JSON"), 
      tags$hr(),
      br(),
      h4("Bulk processing"),
      fileInput(
        "bulk_files",
        "Bulk documents (PDF / PNG / JPEG)",
        multiple = TRUE,
        accept = c(".pdf", ".png", ".jpg", ".jpeg")
      ),
      helpText("Uses the current profile: fields, user prompt, OCR & extraction models, DPI."),
      div(
        class = "d-flex gap-2",
        actionButton(
          "run_bulk_ocr",
          "1) Bulk OCR",
          class = "btn-warning flex-fill"
        ),
        actionButton(
          "run_bulk_extract",
          "2) Bulk extract",
          class = "btn-danger flex-fill"
        )
      ),
      br(), br(),
      downloadButton("download_bulk_csv",  "Download bulk CSV"),
      downloadButton("download_bulk_json", "Download bulk JSON")
    ),

    mainPanel(
      tabsetPanel(
        id = "mode_tabs",
        tabPanel(
          "Single File",
          # --- everything you already had for single-file mode goes here ---
          h4("Original document preview"),
          uiOutput("page_selector_ui"),

          fluidRow(
            column(
              width = 6,
              imageOutput("original_page_image", height = "auto")
            ),
            column(
              width = 6,
              div(
                class = "ocr-page-text-wrapper",
                h5("OCR text for selected page (editable)"),
                textAreaInput(
                  "edit_page_text",
                  label = NULL,
                  value = "",
                  rows  = 24,
                  resize = "vertical",
                  width  = "100%"
                ),
                br(),
                div(
                  class = "d-flex gap-2",
                  actionButton(
                    "save_page_edit",
                    "Save page edit",
                    class = "btn-warning flex-fill"
                  ),
                  actionButton(
                    "reset_page_edit",
                    "Reset to original",
                    class = "btn-secondary flex-fill"
                  )
                ),
                br(),
                # search/replace, diff, etc (what we already added)
                # ...
                tags$small("Edits here will be used for field extraction.")
              )
            )
          ),

          tags$hr(),

          tabsetPanel(
            id = "tabs",
            tabPanel("OCR Text - After Step 1", verbatimTextOutput("ocr_output", placeholder = TRUE)),
            tabPanel("Extracted Fields - After Step 2", tableOutput("extract_table")),
            tabPanel("Raw JSON (debug) - After Step 2", verbatimTextOutput("raw_json_output", placeholder = TRUE))
          )
        ),

        tabPanel(
          "Bulk Mode",
          # --- NEW bulk UI (next section) ---
          h4("Bulk preview & results"),
          fluidRow(
            column(
              width = 6,
              selectInput(
                "bulk_file_select",
                "File",
                choices = character(0)
              )
            ),
            column(
              width = 6,
              uiOutput("bulk_page_selector_ui")
            )
          ),
          fluidRow(
            column(
              width = 6,
              imageOutput("bulk_page_image", height = "auto")
            ),
            column(
              width = 6,
              div(
                class = "ocr-page-text-wrapper",
                h5("Bulk OCR text for selected file & page (editable)"),
                textAreaInput(
                  "bulk_edit_page_text",
                  label = NULL,
                  value = "",
                  rows  = 24,
                  resize = "vertical",
                  width  = "100%"
                ),
                br(),
                div(
                  class = "d-flex gap-2",
                  actionButton(
                    "bulk_save_page_edit",
                    "Save page edit",
                    class = "btn-warning flex-fill"
                  ),
                  actionButton(
                    "bulk_reset_page_edit",
                    "Reset to original",
                    class = "btn-secondary flex-fill"
                  )
                ),
                br(),
                tags$small("Edits here affect bulk extraction for this file.")
              )
            )
          ),

          tags$hr(),
          h4("Aggregated results (fields × files)"),
          tableOutput("bulk_table")
        )
      )
    )

  )
)

# ---- Shiny server --------------------------------------------------

compute_page_diff_html <- function(original, edited) {
  if (is.null(original)) original <- ""
  if (is.null(edited))   edited   <- ""

  orig_lines <- strsplit(original, "\n", fixed = TRUE)[[1]]
  edit_lines <- strsplit(edited,   "\n", fixed = TRUE)[[1]]

  n <- max(length(orig_lines), length(edit_lines))
  out <- character(n)

  for (i in seq_len(n)) {
    o <- if (i <= length(orig_lines)) orig_lines[i] else ""
    e <- if (i <= length(edit_lines)) edit_lines[i] else ""

    if (identical(o, e)) {
      out[i] <- sprintf(
        "<div class='diff-line-same'>%s</div>",
        htmltools::htmlEscape(e)
      )
    } else if (nzchar(e) && !nzchar(o)) {
      out[i] <- sprintf(
        "<div class='diff-line-added'>+ %s</div>",
        htmltools::htmlEscape(e)
      )
    } else if (nzchar(o) && !nzchar(e)) {
      out[i] <- sprintf(
        "<div class='diff-line-removed'>- %s</div>",
        htmltools::htmlEscape(o)
      )
    } else {
      out[i] <- sprintf(
        "<div class='diff-line-modified'>~ %s</div>",
        htmltools::htmlEscape(e)
      )
    }
  }

  paste(out, collapse = "\n")
}

server <- function(input, output, session) {

  # reactive storage
  ocr_result   <- reactiveVal(NULL)  # list(text_per_page, combined_text)
  extract_df   <- reactiveVal(NULL)  # data.frame(name, value, confidence)
  extract_json <- reactiveVal(NULL)  # raw JSON string (for debug/export)
  ocr_running  <- reactiveVal(FALSE) # OCR is running
  edited_page_texts   <- reactiveVal(NULL)  # NEW: character vector per page
  # Bulk mode storage
  bulk_ocr_results    <- reactiveVal(list())  # name -> list(text_per_page, combined_text, page_paths, page_numbers)
  bulk_extract_results <- reactiveVal(list()) # name -> list(df = data.frame(...), raw = model_text)
  bulk_edited_page_texts <- reactiveVal(list())  # list[file_name] -> character vector per page

  example_extract_text <- reactiveVal(NULL)

  observeEvent(input$use_example_extract_text, {
    p <- input$example_extract_text
    if (is.null(p) || !nzchar(p) || !file.exists(p)) {
      showNotification("Example OCR text file not found.", type = "error")
      return(NULL)
    }

    txt <- paste(readLines(p, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
    example_extract_text(txt)

    showNotification(paste("Using example OCR text:", basename(p)), type = "message")
  })

  # Tracks whichever document is "selected" (upload or example)
  current_doc <- reactiveVal(list(path = NULL, name = NULL, source = NULL))

  # When user uploads a file, set it as current doc
  observeEvent(input$file, {
    if (!is.null(input$file) && nrow(input$file) > 0) {
      current_doc(list(
        path = input$file$datapath,
        name = input$file$name,
        source = "upload"
      ))
    }
  }, ignoreInit = TRUE)

  # When user clicks "Use this example", set selected example as current doc
  observeEvent(input$use_example, {
    ex_path <- input$example_doc
    if (is.null(ex_path) || !nzchar(ex_path)) {
      showNotification("No example selected.", type = "warning")
      return(NULL)
    }
    if (!file.exists(ex_path)) {
      showNotification(paste("Example file not found:", ex_path), type = "error", duration = NULL)
      return(NULL)
    }

    current_doc(list(
      path = ex_path,
      name = basename(ex_path),
      source = "example"
    ))

    # Optional: jump to single-file tab so they immediately see preview
    updateTabsetPanel(session, "mode_tabs", selected = "Single File")

    showNotification(paste("Selected example:", basename(ex_path)), type = "message")
  })

  output$selected_doc_info <- renderUI({
    doc <- current_doc()
    if (is.null(doc$path)) {
      return(helpText("No document selected yet. Upload a file or pick an example."))
    }
    tags$small(sprintf("Selected: %s (%s)", doc$name, doc$source))
  })

  run_extraction_on_text <- function(ocr_text_full) {
    fields <- strsplit(input$field_list, "\\r?\\n")[[1]]
    fields <- trimws(fields)
    fields <- fields[nzchar(fields)]

    if (length(fields) == 0) {
      stop("Please provide at least one field name.")
    }

    user_prompt <- input$user_prompt
    model       <- trimws(input$extract_model)

    system_msg <- paste0(
      "You are an information extraction model. ",
      "The user will provide OCR text from a document and a list of target fields. ",
      "Your job is to fill in those fields based on the OCR text and provide a confidence score in [0,1]. ",
      "If a field is missing or not inferable, set value to an empty string and confidence to 0.\n\n",
      "Return ONLY valid JSON with this exact structure:\n",
      "{\n",
      "  \"fields\": [\n",
      "    { \"name\": \"FieldName1\", \"value\": \"string\", \"confidence\": 0.0 },\n",
      "    { \"name\": \"FieldName2\", \"value\": \"string\", \"confidence\": 0.0 }\n",
      "  ]\n",
      "}\n",
      "No markdown, no explanation, no extra text."
    )

    fields_str <- paste0("- ", fields, collapse = "\n")

    # Optional truncation
    ocr_text <- ocr_text_full
    max_chars <- 32000L
    if (!is.null(ocr_text) && nchar(ocr_text) > max_chars) {
      cat("[DEBUG] (bulk) OCR text truncated from", nchar(ocr_text),
          "to", max_chars, "characters\n")
      ocr_text <- substr(ocr_text, 1, max_chars)
    }

    user_msg <- paste0(
      "USER INSTRUCTIONS:\n",
      user_prompt, "\n\n",
      "FIELDS TO EXTRACT:\n",
      fields_str, "\n\n",
      "OCR TEXT (possibly truncated):\n",
      ocr_text
    )

    resp_text <- NULL

    if (identical(model, "gpt-oss")) {
      full_prompt <- paste(
        system_msg,
        "\n\n---\n\n",
        user_msg,
        sep = ""
      )

      resp_text <- generate(
        model       = model,
        prompt      = full_prompt,
        stream      = FALSE,
        output      = "text",
        temperature = 0
      )
    } else {
      resp_text <- chat(
        model    = model,
        messages = list(
          list(role = "system", content = system_msg),
          list(role = "user",   content = user_msg)
        ),
        stream      = FALSE,
        output      = "text",
        temperature = 0
      )
    }

    if (is.null(resp_text) || !nzchar(resp_text)) {
      stop("Extraction model returned empty or NULL response.")
    }

    json_str <- extract_json_substring(resp_text)
    parsed   <- jsonlite::fromJSON(json_str, simplifyVector = TRUE)

    if (is.null(parsed$fields)) {
      stop("Response JSON does not contain 'fields' key.")
    }

    df <- as.data.frame(parsed$fields, stringsAsFactors = FALSE)
    names(df) <- tolower(names(df))
    if (!"name" %in% names(df)) df$name <- NA_character_
    if (!"value" %in% names(df)) df$value <- NA_character_
    if (!"confidence" %in% names(df)) df$confidence <- NA_real_

    list(df = df, raw = resp_text)
  }

  spellcheck_pages <- function(pages) {
    # pages: character vector, one entry per page
    if (!requireNamespace("hunspell", quietly = TRUE)) {
      showNotification(
        "Package 'hunspell' not installed; skipping spellcheck.",
        type = "warning"
      )
      return(pages)
    }

    out <- pages
    for (i in seq_along(pages)) {
      txt <- pages[[i]]
      if (is.null(txt) || !nzchar(txt)) next

      # Find misspelled words
      bad_list <- hunspell::hunspell(txt)[[1]]
      if (length(bad_list) == 0) next

      for (w in bad_list) {
        sugg <- hunspell::hunspell_suggest(w)[[1]]
        # Only auto-replace if there's exactly one suggestion
        if (length(sugg) == 1) {
          # Simple word boundary replace
          txt <- gsub(
            pattern = paste0("\\b", w, "\\b"),
            replacement = sugg[1],
            x = txt,
            perl = TRUE
          )
        }
      }
      out[[i]] <- txt
    }

    out
  }

  # --- Dark mode handling ------------------------------------------
  # 1) On load, check system dark preference (sent from JS) and set theme once
  observeEvent(input$system_pref_dark, {
    if (isTRUE(input$system_pref_dark)) {
      session$setCurrentTheme(dark_theme)
      updateCheckboxInput(session, "dark_mode", value = TRUE)
    }
  }, once = TRUE)

  # 2) Manual toggle via checkbox
  observeEvent(input$dark_mode, {
    if (isTRUE(input$dark_mode)) {
      session$setCurrentTheme(dark_theme)
    } else {
      session$setCurrentTheme(light_theme)
    }
  })

  # --- Update editable page text area when page index or OCR changes ----
  observeEvent(
    list(input$page_index, ocr_result(), edited_page_texts()),
    {
      res <- ocr_result()
      edt <- edited_page_texts()

      if (is.null(res) || is.null(res$text_per_page)) {
        updateTextAreaInput(session, "edit_page_text", value = "")
        return(NULL)
      }

      n_pages <- length(res$text_per_page)
      idx <- 1L
      if (!is.null(input$page_index) && !is.na(input$page_index)) {
        idx <- as.integer(input$page_index)
        if (idx < 1L || idx > n_pages) idx <- 1L
      }

      # use edited version if available, else raw OCR for that page
      page_text <- if (!is.null(edt) && length(edt) >= idx) {
        edt[[idx]]
      } else {
        res$text_per_page[[idx]]
      }

      updateTextAreaInput(session, "edit_page_text", value = page_text)
    },
    ignoreInit = FALSE
  )

  # --- Reset page edit to original OCR -----------------------------
  observeEvent(input$reset_page_edit, {
    res <- ocr_result()
    if (is.null(res) || is.null(res$text_per_page)) {
      showNotification("No OCR result to reset.", type = "warning")
      return(NULL)
    }

    n_pages <- length(res$text_per_page)

    idx <- 1L
    if (!is.null(input$page_index) && !is.na(input$page_index)) {
      idx <- as.integer(input$page_index)
      if (idx < 1L || idx > n_pages) {
        idx <- 1L
      }
    }

    # Get original OCR text for this page
    original_txt <- res$text_per_page[[idx]]

    # Reset this page in the edited vector
    edt <- edited_page_texts()
    if (is.null(edt) || length(edt) != n_pages) {
      edt <- res$text_per_page
    }
    edt[[idx]] <- original_txt
    edited_page_texts(edt)

    # Update the text area to match original
    updateTextAreaInput(session, "edit_page_text", value = original_txt)

    showNotification(
      paste("Reset page", idx, "to original OCR text."),
      type = "message"
    )
  })

  # --- Save edited page text ---------------------------------------
  observeEvent(input$save_page_edit, {
    res <- ocr_result()
    if (is.null(res) || is.null(res$text_per_page)) {
      showNotification("No OCR result to edit yet.", type = "warning")
      return(NULL)
    }

    n_pages <- length(res$text_per_page)
    idx <- 1L
    if (!is.null(input$page_index) && !is.na(input$page_index)) {
      idx <- as.integer(input$page_index)
      if (idx < 1L || idx > n_pages) idx <- 1L
    }

    txt <- input$edit_page_text
    if (is.null(txt)) txt <- ""

    edt <- edited_page_texts()
    if (is.null(edt) || length(edt) != n_pages) {
      edt <- res$text_per_page
    }

    edt[[idx]] <- txt
    edited_page_texts(edt)

    showNotification(
      paste("Saved edits for page", idx),
      type = "message"
    )
  })
  # --- Compute and render page diff -------------------------------
  output$page_diff <- renderUI({
    if (!isTRUE(input$show_page_diff)) {
      return(NULL)
    }

    res <- ocr_result()
    edt <- edited_page_texts()

    if (is.null(res) || is.null(res$text_per_page)) {
      return(HTML("<em>No OCR text available for diff.</em>"))
    }

    n_pages <- length(res$text_per_page)
    idx <- 1L
    if (!is.null(input$page_index) && !is.na(input$page_index)) {
      idx <- as.integer(input$page_index)
      if (idx < 1L || idx > n_pages) idx <- 1L
    }

    original_txt <- res$text_per_page[[idx]]
    edited_txt <- if (!is.null(edt) && length(edt) >= idx) {
      edt[[idx]]
    } else {
      original_txt
    }

    HTML(compute_page_diff_html(original_txt, edited_txt))
  })


  # # --- Save edited OCR text ----------------------------------------
  # observeEvent(input$save_edited_ocr, {
  #   txt <- input$edit_ocr_text
  #   if (is.null(txt) || !nzchar(txt)) {
  #     showNotification(
  #       "Edited OCR text is empty. Not saving.",
  #       type = "warning"
  #     )
  #     return(NULL)
  #   }

  #   edited_ocr_text(txt)
  #   showNotification(
  #     "Edited OCR text saved. Field extraction will now use this version.",
  #     type = "message"
  #   )
  # })
  # --- Bulk OCR + extraction --------------------------------------
observeEvent(input$run_bulk_ocr, {
  files <- input$bulk_files
  req(files)

  # Reset previous OCR + extraction
  bulk_ocr_results(list())
  bulk_extract_results(list())
  bulk_edited_page_texts(list())

  withProgress(
    message = "Running bulk OCR...",
    value = 0,
    {
      n_files <- nrow(files)
      ocr_res_list <- list()
      edt_list     <- list()

      for (i in seq_len(n_files)) {
        incProgress(
          amount = 1 / n_files,
          detail = sprintf("File %d of %d: %s", i, n_files, files$name[i])
        )

        file_path <- files$datapath[i]
        fname     <- files$name[i]

        ocr_res <- tryCatch(
          {
            ocr_any_file_ollama(
              file_path = file_path,
              model     = input$ocr_model,
              dpi       = input$dpi,
              prompt    = "Transcribe all text in this scanned page as plain UTF-8 text, preserving reading order.",
              progress_callback = NULL
            )
          },
          error = function(e) {
            showNotification(
              paste("Bulk OCR error for", fname, ":", e$message),
              type = "error"
            )
            NULL
          }
        )

        if (is.null(ocr_res)) next

        ocr_res_list[[fname]] <- ocr_res

        if (!is.null(ocr_res$text_per_page)) {
          edt_list[[fname]] <- ocr_res$text_per_page
        }
      }

      bulk_ocr_results(ocr_res_list)
      bulk_edited_page_texts(edt_list)

      if (length(ocr_res_list) > 0) {
        updateSelectInput(
          session,
          "bulk_file_select",
          choices = names(ocr_res_list),
          selected = names(ocr_res_list)[1]
        )
      }
    }
  )

  # If you have a top-level tabset for modes:
  # updateTabsetPanel(session, "mode_tabs", selected = "Bulk Mode")
  })

observeEvent(input$run_bulk_extract, {
  ocr_list <- bulk_ocr_results()
  req(length(ocr_list) > 0)

  withProgress(
    message = "Running bulk extraction...",
    value = 0,
    {
      file_names <- names(ocr_list)
      n_files <- length(file_names)
      ext_res_list <- list()
      edt_list <- bulk_edited_page_texts()

      for (i in seq_len(n_files)) {
        fname <- file_names[i]
        incProgress(
          amount = 1 / n_files,
          detail = sprintf("File %d of %d: %s", i, n_files, fname)
        )

        ocr_res <- ocr_list[[fname]]
        if (is.null(ocr_res$text_per_page)) next

        # Use edited pages if available, otherwise original OCR
        page_vec <- if (!is.null(edt_list) &&
                        !is.null(edt_list[[fname]]) &&
                        length(edt_list[[fname]]) == length(ocr_res$text_per_page)) {
          edt_list[[fname]]
        } else {
          ocr_res$text_per_page
        }

        page_nums <- if (!is.null(ocr_res$page_numbers)) {
          ocr_res$page_numbers
        } else {
          seq_along(page_vec)
        }

        combined_text <- paste(
          sprintf("---- PAGE %d ----\n%s", page_nums, page_vec),
          collapse = "\n\n"
        )

        ext_res <- tryCatch(
          {
            run_extraction_on_text(combined_text)
          },
          error = function(e) {
            showNotification(
              paste("Bulk extraction error for", fname, ":", e$message),
              type = "error"
            )
            NULL
          }
        )

        if (!is.null(ext_res)) {
          ext_res_list[[fname]] <- ext_res
        }
      }

      bulk_extract_results(ext_res_list)
    }
  )
  })

  # --- Bulk page image viewer -------------------------------------
  output$bulk_page_selector_ui <- renderUI({
    ocr_list <- bulk_ocr_results()
    fname <- input$bulk_file_select

    if (is.null(fname) || !nzchar(fname) || is.null(ocr_list[[fname]])) {
      return(helpText("Run bulk OCR + extraction, then select a file."))
    }

    res <- ocr_list[[fname]]
    if (is.null(res$text_per_page)) {
      return(helpText("No per-page OCR text available for this file."))
    }

    n_pages <- length(res$text_per_page)

    if (n_pages <= 1) {
      return(tags$strong("Single-page document"))
    }

    selectInput(
      "bulk_page_index",
      "Page:",
      choices  = seq_len(n_pages),
      selected = 1
    )
  })

  
  output$bulk_page_image <- renderImage({
    ocr_list <- bulk_ocr_results()
    fname <- input$bulk_file_select

    if (is.null(fname) || !nzchar(fname) || is.null(ocr_list[[fname]])) {
      return(list(src = "", contentType = NULL, alt = "No bulk preview yet."))
    }

    res <- ocr_list[[fname]]
    if (is.null(res$page_paths)) {
      return(list(src = "", contentType = NULL, alt = "No page images available."))
    }

    n_pages <- length(res$page_paths)
    idx <- 1L
    if (!is.null(input$bulk_page_index) && !is.na(input$bulk_page_index)) {
      idx <- as.integer(input$bulk_page_index)
      if (idx < 1L || idx > n_pages) idx <- 1L
    }

    img_path <- res$page_paths[idx]
    ext <- tolower(tools::file_ext(img_path))
    ctype <- switch(
      ext,
      "png"  = "image/png",
      "jpg"  = "image/jpeg",
      "jpeg" = "image/jpeg",
      "image/png"
    )

    list(
      src         = img_path,
      contentType = ctype,
      alt         = paste("File", fname, "- page", idx)
    )
  }, deleteFile = FALSE)


  output$bulk_page_text <- renderText({
    ocr_list <- bulk_ocr_results()
    fname <- input$bulk_file_select

    if (is.null(fname) || !nzchar(fname) || is.null(ocr_list[[fname]])) {
      return("No bulk OCR result yet for this file.")
    }

    res <- ocr_list[[fname]]
    if (is.null(res$text_per_page)) {
      return("No per-page OCR text available for this file.")
    }

    n_pages <- length(res$text_per_page)
    idx <- 1L
    if (!is.null(input$bulk_page_index) && !is.na(input$bulk_page_index)) {
      idx <- as.integer(input$bulk_page_index)
      if (idx < 1L || idx > n_pages) idx <- 1L
    }

    page_num <- if (!is.null(res$page_numbers)) res$page_numbers[idx] else idx

    paste0(
      sprintf("---- FILE: %s | PAGE %d ----\n", fname, page_num),
      res$text_per_page[[idx]]
    )
  })

  bulk_table_reactive <- reactive({
    ext_list <- bulk_extract_results()
    if (!length(ext_list)) return(NULL)

    # Collect all unique field names
    all_fields <- sort(unique(unlist(lapply(ext_list, function(x) x$df$name))))

    tab <- data.frame(Field = all_fields, stringsAsFactors = FALSE)

    for (fname in names(ext_list)) {
      df <- ext_list[[fname]]$df
      # match field order
      tab[[fname]] <- df$value[match(all_fields, df$name)]
    }

    tab
  })
  
  output$bulk_table <- renderTable({
    bulk_table_reactive()
  }, bordered = TRUE, striped = TRUE, na = "")

  output$download_bulk_csv <- downloadHandler(
    filename = function() {
      paste0("bulk_results_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv")
    },
    content = function(file) {
      tab <- bulk_table_reactive()
      if (is.null(tab)) {
        write.csv(data.frame(), file, row.names = FALSE)
      } else {
        write.csv(tab, file, row.names = FALSE)
      }
    }
  )

  output$download_bulk_json <- downloadHandler(
    filename = function() {
      paste0("bulk_results_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".json")
    },
    content = function(file) {
      tab <- bulk_table_reactive()
      if (is.null(tab)) {
        writeLines("{}", con = file)
        return()
      }

      field_names <- tab$Field
      files <- setdiff(names(tab), "Field")

      out_list <- setNames(
        lapply(seq_along(field_names), function(i) {
          vals <- as.list(tab[i, files, drop = FALSE])
          # drop NAs
          vals <- lapply(vals, function(x) if (is.na(x)) "" else x)
          vals
        }),
        field_names
      )

      jsonlite::write_json(out_list, path = file, pretty = TRUE, auto_unbox = TRUE)
    }
  )

  # --- Page selector for original viewer ---------------------------
  output$page_selector_ui <- renderUI({
    res <- ocr_result()
    if (is.null(res) || is.null(res$text_per_page)) {
      return(helpText("Run OCR to enable page preview."))
    }

    n_pages <- length(res$text_per_page)

    if (n_pages <= 1) {
      return(tags$strong("Single-page document"))
    }

    selectInput(
      "page_index",
      "Page to display:",
      choices  = seq_len(n_pages),
      selected = 1
    )
  })

  output$ocr_page_text <- renderText({
    res <- ocr_result()
    if (is.null(res) || is.null(res$text_per_page)) {
      return("Run OCR to see per-page text here.")
    }

    n_pages <- length(res$text_per_page)
    idx <- 1L

    if (!is.null(input$page_index) && !is.na(input$page_index)) {
      idx <- as.integer(input$page_index)
      if (idx < 1L || idx > n_pages) {
        idx <- 1L
      }
    }

    # Use real page number if available, else fall back to index
    page_num <- if (!is.null(res$page_numbers)) res$page_numbers[idx] else idx

    paste0(
      sprintf("---- PAGE %d ----\n", page_num),
      res$text_per_page[idx]
    )
  })

  # --- Original page image (using OCR-generated PNGs) --------------
  output$original_page_image <- renderImage({
    res <- ocr_result()
    if (is.null(res) || is.null(res$page_paths)) {
      # Nothing yet; Shiny needs a list
      return(list(
        src         = "",
        contentType = NULL,
        alt         = "No document preview yet."
      ))
    }

    n_pages <- length(res$page_paths)
    page_index <- 1L

    if (!is.null(input$page_index) && !is.na(input$page_index)) {
      page_index <- as.integer(input$page_index)
      if (page_index < 1L || page_index > n_pages) {
        page_index <- 1L
      }
    }

    img_path <- res$page_paths[page_index]
    ext <- tolower(tools::file_ext(img_path))
    ctype <- switch(
      ext,
      "png"  = "image/png",
      "jpg"  = "image/jpeg",
      "jpeg" = "image/jpeg",
      "image/png"
    )

    list(
      src         = img_path,
      contentType = ctype,
      alt         = paste("Page", page_index)
    )
  }, deleteFile = FALSE)


  output$ocr_spinner <- renderUI({
    if (!isTRUE(ocr_running())) {
      return(div(class = "ocr-spinner-container"))  # empty but keeps layout stable
    }

    div(
      class = "ocr-spinner-container",
      tags$div(
        class = "spinner-border text-secondary",
        role  = "status",
        tags$span(class = "visually-hidden", "Running OCR...\n")
      )
    )
  })

  # Initialize profile list in UI
  observe({
    updateSelectInput(
      session,
      "load_profile_name",
      choices = list_profiles()
    )
  })
  # --- Replace all occurrences on this page -----------------------
  observeEvent(input$replace_all, {
    txt <- input$edit_page_text
    pattern <- input$search_text
    repl <- input$replace_text

    if (is.null(txt)) {
      showNotification("No editable text available for this page.", type = "warning")
      return(NULL)
    }
    if (is.null(pattern) || !nzchar(pattern)) {
      showNotification("Enter some text in 'Find' to search for.", type = "warning")
      return(NULL)
    }

    # Literal search (no regex) to avoid surprises
    new_txt <- gsub(pattern, repl, txt, fixed = TRUE)

    if (identical(new_txt, txt)) {
      showNotification("No matches found on this page.", type = "message")
      return(NULL)
    }

    # Update the text area only; user still clicks "Save page edit" to commit
    updateTextAreaInput(session, "edit_page_text", value = new_txt)

    showNotification("Replaced all occurrences on this page. Click 'Save page edit' to commit.", type = "message")
  })

  # ---- OCR step ----------------------------------------------------


  observeEvent(
    list(input$bulk_file_select, input$bulk_page_index, bulk_ocr_results(), bulk_edited_page_texts()),
    {
      ocr_list <- bulk_ocr_results()
      edt_list <- bulk_edited_page_texts()
      fname <- input$bulk_file_select

      if (is.null(fname) || !nzchar(fname) || is.null(ocr_list[[fname]])) {
        updateTextAreaInput(session, "bulk_edit_page_text", value = "")
        return(NULL)
      }

      res <- ocr_list[[fname]]
      if (is.null(res$text_per_page)) {
        updateTextAreaInput(session, "bulk_edit_page_text", value = "")
        return(NULL)
      }

      n_pages <- length(res$text_per_page)
      idx <- 1L
      if (!is.null(input$bulk_page_index) && !is.na(input$bulk_page_index)) {
        idx <- as.integer(input$bulk_page_index)
        if (idx < 1L || idx > n_pages) idx <- 1L
      }

      # choose edited text if available
      page_vec <- if (!is.null(edt_list[[fname]]) &&
                      length(edt_list[[fname]]) == n_pages) {
        edt_list[[fname]]
      } else {
        res$text_per_page
      }

      updateTextAreaInput(session, "bulk_edit_page_text", value = page_vec[[idx]])
    },
    ignoreInit = TRUE
  )
  # --- Reset bulk page edit to original OCR -----------------------

  observeEvent(input$bulk_save_page_edit, {
    ocr_list <- bulk_ocr_results()
    edt_list <- bulk_edited_page_texts()
    fname <- input$bulk_file_select

    if (is.null(fname) || !nzchar(fname) || is.null(ocr_list[[fname]])) {
      showNotification("No bulk file selected to save edits for.", type = "warning")
      return(NULL)
    }

    res <- ocr_list[[fname]]
    if (is.null(res$text_per_page)) {
      showNotification("No OCR text to edit for this file.", type = "warning")
      return(NULL)
    }

    n_pages <- length(res$text_per_page)
    idx <- 1L
    if (!is.null(input$bulk_page_index) && !is.na(input$bulk_page_index)) {
      idx <- as.integer(input$bulk_page_index)
      if (idx < 1L || idx > n_pages) idx <- 1L
    }

    txt <- input$bulk_edit_page_text
    if (is.null(txt)) txt <- ""

    if (is.null(edt_list[[fname]]) || length(edt_list[[fname]]) != n_pages) {
      edt_list[[fname]] <- res$text_per_page
    }

    edt_vec <- edt_list[[fname]]
    edt_vec[[idx]] <- txt
    edt_list[[fname]] <- edt_vec

    bulk_edited_page_texts(edt_list)

    showNotification(
      paste("Saved edits for file", fname, "- page", idx),
      type = "message"
    )
  })

  observeEvent(input$bulk_reset_page_edit, {
    ocr_list <- bulk_ocr_results()
    edt_list <- bulk_edited_page_texts()
    fname <- input$bulk_file_select

    if (is.null(fname) || !nzchar(fname) || is.null(ocr_list[[fname]])) {
      showNotification("No bulk file selected to reset.", type = "warning")
      return(NULL)
    }

    res <- ocr_list[[fname]]
    if (is.null(res$text_per_page)) {
      showNotification("No OCR text to reset for this file.", type = "warning")
      return(NULL)
    }

    n_pages <- length(res$text_per_page)
    idx <- 1L
    if (!is.null(input$bulk_page_index) && !is.na(input$bulk_page_index)) {
      idx <- as.integer(input$bulk_page_index)
      if (idx < 1L || idx > n_pages) idx <- 1L
    }

    original <- res$text_per_page[[idx]]

    if (is.null(edt_list[[fname]]) || length(edt_list[[fname]]) != n_pages) {
      edt_list[[fname]] <- res$text_per_page
    }

    edt_vec <- edt_list[[fname]]
    edt_vec[[idx]] <- original
    edt_list[[fname]] <- edt_vec

    bulk_edited_page_texts(edt_list)
    updateTextAreaInput(session, "bulk_edit_page_text", value = original)

    showNotification(
      paste("Reset file", fname, "- page", idx, "to original OCR."),
      type = "message"
    )
  })


  observeEvent(input$run_ocr, {
    doc <- current_doc()
    if (is.null(doc$path) || !file.exists(doc$path)) {
      showNotification("No document selected (or file missing). Upload a file or choose an example.", type = "warning")
      return(NULL)
    }

    # reset prior results
    ocr_result(NULL)
    extract_df(NULL)
    extract_json(NULL)
    edited_page_texts(NULL)

    file_path <- doc$path
    dpi       <- input$dpi
    model     <- input$ocr_model

    ocr_running(TRUE)  # <<< START SPINNER

    withProgress(
      message = "Running OCR with Ollama...\n",
      value = 0,
      {
        # ensure we always turn it off at the end, even on error
        on.exit(ocr_running(FALSE), add = TRUE)  # <<< STOP SPINNER WHEN DONE

        progress_cb <- function(i, n) {
          incProgress(1 / n, detail = sprintf("Page %d of %d", i, n))
        }

        res <- tryCatch(
          {
            ext <- tolower(tools::file_ext(file_path))

            pages_to_ocr <- NULL

            if (ext == "pdf") {
              info <- pdftools::pdf_info(file_path)
              n_total <- info$pages

              # Resolve start page
              if (is.null(input$page_start) || is.na(input$page_start)) {
                p_start <- 1L
              } else {
                p_start <- max(1L, as.integer(input$page_start))
              }

              # Resolve end page
              if (is.null(input$page_end) || is.na(input$page_end)) {
                p_end <- n_total
              } else {
                p_end <- min(n_total, as.integer(input$page_end))
              }

              if (p_start > p_end) {
                showNotification(
                  "Page range invalid (start > end). Using all pages instead.",
                  type = "warning"
                )
                pages_to_ocr <- seq_len(n_total)
              } else {
                pages_to_ocr <- seq(p_start, p_end)
              }
            }

            ocr_any_file_ollama(
              file_path = file_path,
              model     = model,
              dpi       = dpi,
              prompt    = "Transcribe all text in this scanned page as plain UTF-8 text, preserving reading order.",
              progress_callback = progress_cb,
              pages     = pages_to_ocr
            )
          },
          error = function(e) {
            showNotification(
              paste("Error during OCR:", e$message),
              type = "error"
            )
            NULL
          }
        )

        ocr_result(res)
        if (!is.null(res) && !is.null(res$text_per_page)) {
          # initialize editable texts as a copy of OCR output
          edited_page_texts(res$text_per_page)
        }      
      }
    )
  })

  # ---- Field extraction step --------------------------------------

  observeEvent(input$run_extract, {
    # Optional uploaded OCR text (bypasses need for run_ocr)
    uploaded_text <- NULL
    if (!is.null(input$extract_text_file) && nrow(input$extract_text_file) > 0) {
      uploaded_text <- paste(
        readLines(input$extract_text_file$datapath, warn = FALSE, encoding = "UTF-8"),
        collapse = "\n"
      )
    }

    sample_text <- example_extract_text()

    # Pick the best available text source
    chosen_text <- NULL
    if (!is.null(uploaded_text) && nzchar(uploaded_text)) {
      chosen_text <- uploaded_text
    } else if (!is.null(sample_text) && nzchar(sample_text)) {
      chosen_text <- sample_text
    } else {
      chosen_text <- NULL
    }

    res <- ocr_result()

    # If no chosen_text, require OCR result as fallback
    if (is.null(chosen_text) || !nzchar(chosen_text)) {
      req(res)
    }

    # Fields
    fields <- strsplit(input$field_list, "\\r?\\n")[[1]]
    fields <- trimws(fields)
    fields <- fields[nzchar(fields)]  # drop empty

    if (length(fields) == 0) {
      showNotification("Please provide at least one field name.", type = "warning")
      return(NULL)
    }

    user_prompt <- input$user_prompt
    fields_str  <- paste0("- ", fields, collapse = "\n")

    withProgress(
      message = "Running field extraction with gpt-oss...",
      value = 0,
      {
        incProgress(0.2, detail = "Calling Ollama…")

        df <- NULL
        resp_text <- NULL

        df <- tryCatch(
          {
            system_msg <- paste0(
              "You are an information extraction model. ",
              "The user will provide OCR text from a document and a list of target fields. ",
              "Your job is to fill in those fields based on the OCR text and provide a confidence score in [0,1]. ",
              "If a field is missing or not inferable, set value to an empty string and confidence to 0.\n\n",
              "Return ONLY valid JSON with this exact structure:\n",
              "{\n",
              "  \"fields\": [\n",
              "    { \"name\": \"FieldName1\", \"value\": \"string\", \"confidence\": 0.0 },\n",
              "    { \"name\": \"FieldName2\", \"value\": \"string\", \"confidence\": 0.0 }\n",
              "  ]\n",
              "}\n",
              "No markdown, no explanation, no extra text."
            )

            # ---- Decide which OCR text to use ----
            if (!is.null(chosen_text) && nzchar(chosen_text)) {
              # Use user-uploaded OCR text directly
              ocr_text <- chosen_text
            } else {
              # Build combined text from per-page edits if present
              edt <- edited_page_texts()
              page_nums <- if (!is.null(res$page_numbers)) {
                res$page_numbers
              } else if (!is.null(res$text_per_page)) {
                seq_along(res$text_per_page)
              } else {
                NULL
              }

              # Optional spellcheck on per-page text
              if (!is.null(edt) && isTRUE(input$auto_spellcheck)) {
                edt_checked <- spellcheck_pages(edt)
              } else {
                edt_checked <- edt
              }

              if (!is.null(edt_checked) && !is.null(page_nums) &&
                  length(edt_checked) == length(page_nums)) {
                base_text <- paste(
                  sprintf("---- PAGE %d ----\n%s", page_nums, edt_checked),
                  collapse = "\n\n"
                )
              } else {
                base_text <- res$combined_text
              }

              ocr_text <- base_text
            }

            # Optional truncation (protects the model)
            max_chars <- 32000L
            if (!is.null(ocr_text) && nchar(ocr_text) > max_chars) {
              cat("[DEBUG] OCR text truncated from", nchar(ocr_text),
                  "to", max_chars, "characters; using edited/combined OCR.\n")
              ocr_text <- substr(ocr_text, 1, max_chars)
            }

            user_msg <- paste0(
              "USER INSTRUCTIONS:\n",
              user_prompt, "\n\n",
              "FIELDS TO EXTRACT:\n",
              fields_str, "\n\n",
              "OCR TEXT (possibly truncated):\n",
              ocr_text
            )

            cat("=== SYSTEM MESSAGE ===\n")
            cat(system_msg, "\n\n")

            cat("=== USER MESSAGE ===\n")
            cat(substr(user_msg, 1, 1000), "\n\n")

            model <- trimws(input$extract_model)

            if (identical(model, "gpt-oss")) {
              # completion-style prompt
              full_prompt <- paste(
                system_msg,
                "\n\n---\n\n",
                user_msg,
                sep = ""
              )

              resp_text <- generate(
                model       = model,
                prompt      = full_prompt,
                stream      = FALSE,
                output      = "text",
                temperature = 0
              )
            } else {
              # chat-style prompt
              resp_text <- chat(
                model    = model,
                messages = list(
                  list(role = "system", content = system_msg),
                  list(role = "user",   content = user_msg)
                ),
                stream      = FALSE,
                output      = "text",
                temperature = 0
              )
            }

            if (is.null(resp_text) || !nzchar(resp_text)) {
              cat("[DEBUG] Extraction model returned empty or NULL response.\n")
              showNotification(
                "Extraction model returned an empty response. Try again or change the extraction model.",
                type = "error",
                duration = NULL
              )
              return(NULL)
            }

            cat("[DEBUG] First 200 chars of extraction response:\n",
                substr(resp_text, 1, 200), "\n")

            json_str <- extract_json_substring(resp_text)
            parsed   <- jsonlite::fromJSON(json_str, simplifyVector = TRUE)

            if (is.null(parsed$fields)) {
              stop("Response JSON does not contain 'fields' key.")
            }

            df <- as.data.frame(parsed$fields, stringsAsFactors = FALSE)
            names(df) <- tolower(names(df))
            if (!"name" %in% names(df)) stop("Parsed fields must include 'name'.")
            if (!"value" %in% names(df)) df$value <- NA_character_
            if (!"confidence" %in% names(df)) df$confidence <- NA_real_

            df
          },
          error = function(e) {
            showNotification(
              paste("Error during extraction:", e$message),
              type = "error"
            )
            NULL
          }
        )

        if (!is.null(df) && !is.null(resp_text)) {
          extract_df(df)
          extract_json(resp_text)
        }
      }
    )
    updateTabsetPanel(session, "tabs", selected = "Extracted Fields")
  })

  # ---- Save profile ------------------------------------------------

  observeEvent(input$save_profile, {
    name <- input$profile_name
    if (is.null(name) || trimws(name) == "") {
      showNotification("Please enter a profile name before saving.", type = "warning")
      return(NULL)
    }

    tryCatch(
      {
        prof_name <- save_profile_to_disk(name, input)
        showNotification(paste("Profile saved:", prof_name), type = "message")

        # Refresh profile list in UI
        updateSelectInput(
          session,
          "load_profile_name",
          choices = list_profiles(),
          selected = prof_name
        )
      },
      error = function(e) {
        showNotification(
          paste("Error saving profile:", e$message),
          type = "error"
        )
      }
    )
  })

  # ---- Delete profile ----------------------------------------------

  observeEvent(input$delete_profile, {
    prof_name <- input$load_profile_name
    if (is.null(prof_name) || prof_name == "") {
      showNotification("No profile selected to delete.", type = "warning")
      return(NULL)
    }

    tryCatch(
      {
        delete_profile_from_disk(prof_name)
        showNotification(paste("Profile deleted:", prof_name), type = "message")

        # Refresh list and clear selection
        new_choices <- list_profiles()
        updateSelectInput(
          session,
          "load_profile_name",
          choices = new_choices,
          selected = if (length(new_choices) > 0) new_choices[1] else ""
        )
      },
      error = function(e) {
        showNotification(
          paste("Error deleting profile:", e$message),
          type = "error"
        )
      }
    )
  })

  # ---- Load profile ------------------------------------------------

  observeEvent(input$load_profile, {
    prof_name <- input$load_profile_name
    if (is.null(prof_name) || prof_name == "") {
      showNotification("No profile selected to load.", type = "warning")
      return(NULL)
    }

    prof <- tryCatch(
      {
        load_profile_from_disk(prof_name)
      },
      error = function(e) {
        showNotification(
          paste("Error loading profile:", e$message),
          type = "error"
        )
        NULL
      }
    )
    if (is.null(prof)) return(NULL)

    if (!is.null(prof$field_list)) {
      updateTextAreaInput(session, "field_list", value = prof$field_list)
    }
    if (!is.null(prof$user_prompt)) {
      updateTextAreaInput(session, "user_prompt", value = prof$user_prompt)
    }
    if (!is.null(prof$ocr_model)) {
      updateTextInput(session, "ocr_model", value = prof$ocr_model)
    }
    if (!is.null(prof$extract_model)) {
      updateTextInput(session, "extract_model", value = prof$extract_model)
    }
    if (!is.null(prof$dpi)) {
      updateSliderInput(session, "dpi", value = prof$dpi)
    }

    showNotification(paste("Profile loaded:", prof_name), type = "message")
  })

  # ---- Outputs -----------------------------------------------------

  output$ocr_output <- renderText({
    res <- ocr_result()
    if (is.null(res)) {
      return("Upload a file and click 'Run OCR' to see results here.")
    }

    edt <- edited_page_texts()
    page_nums <- if (!is.null(res$page_numbers)) {
      res$page_numbers
    } else if (!is.null(res$text_per_page)) {
      seq_along(res$text_per_page)
    } else {
      NULL
    }

    if (!is.null(edt) && !is.null(page_nums) && length(edt) == length(page_nums)) {
      paste(
        sprintf("---- PAGE %d ----\n%s", page_nums, edt),
        collapse = "\n\n"
      )
    } else {
      res$combined_text
    }
  })

  output$extract_table <- renderTable({
    df <- extract_df()
    if (is.null(df)) return(NULL)
    df
  })

  output$raw_json_output <- renderText({
    txt <- extract_json()
    if (is.null(txt)) {
      return("No extraction run yet, or model did not return text.")
    }
    txt
  })

  # Download CSV
  output$download_csv <- downloadHandler(
    filename = function() {
      doc <- current_doc()
      base <- if (!is.null(doc$name) && nzchar(doc$name)) tools::file_path_sans_ext(doc$name) else "ocr_extraction"
      paste0(base, "_fields.csv")
    },
    content = function(file) {
      df <- extract_df()
      if (is.null(df)) {
        write.csv(data.frame(), file, row.names = FALSE)
      } else {
        write.csv(df, file, row.names = FALSE)
      }
    }
  )

  # Download JSON (field name as key, with value + confidence)
  output$download_json <- downloadHandler(
    filename = function() {
      doc <- current_doc()
      base <- if (!is.null(doc$name) && nzchar(doc$name)) tools::file_path_sans_ext(doc$name) else "ocr_extraction"
      paste0(base, "_fields.json")
    },
    content = function(file) {
      df <- extract_df()
      if (is.null(df)) {
        writeLines("{}", con = file)
      } else {
        out_list <- lapply(seq_len(nrow(df)), function(i) {
          list(
            value      = df$value[i],
            confidence = df$confidence[i]
          )
        })
        names(out_list) <- df$name
        jsonlite::write_json(out_list, path = file, pretty = TRUE, auto_unbox = TRUE)
      }
    }
  )

  # Download OCR text (uses edited per-page text if present)
  output$download_ocr_text <- downloadHandler(
    filename = function() {
      doc <- current_doc()
      base <- if (!is.null(doc$name) && nzchar(doc$name)) tools::file_path_sans_ext(doc$name) else "ocr_extraction"
      paste0(base, "_ocr.txt")
    },
    content = function(file) {
      res <- ocr_result()
      if (is.null(res)) {
        # No OCR done yet
        writeLines("No OCR result available.", con = file, useBytes = TRUE)
        return()
      }

      edt <- edited_page_texts()
      page_nums <- if (!is.null(res$page_numbers)) {
        res$page_numbers
      } else if (!is.null(res$text_per_page)) {
        seq_along(res$text_per_page)
      } else {
        NULL
      }

      if (!is.null(edt) && !is.null(page_nums) &&
          length(edt) == length(page_nums)) {
        txt <- paste(
          sprintf("---- PAGE %d ----\n%s", page_nums, edt),
          collapse = "\n\n"
        )
      } else {
        txt <- res$combined_text
      }

      writeLines(txt, con = file, useBytes = TRUE)
    }
  )


  # Editable OCR text UI
  output$ocr_edit_ui <- renderUI({
    res <- ocr_result()

    # Decide what to show:
    # 1) edited text if present; 2) raw combined OCR; 3) placeholder message
    text_val <- NULL
    if (!is.null(edited_ocr_text())) {
      text_val <- edited_ocr_text()
    } else if (!is.null(res) && !is.null(res$combined_text)) {
      text_val <- res$combined_text
    } else {
      text_val <- "Upload a file and click 'Run OCR' to see and edit text here."
    }

    tagList(
      h5("Editable OCR text"),
      helpText("You can correct handwriting / OCR errors here. Click 'Save edited OCR' before running field extraction."),
      textAreaInput(
        "edit_ocr_text",
        label = NULL,
        value = text_val,
        rows  = 20,
        resize = "vertical",
        width = "100%"
      ),
      actionButton(
        "save_edited_ocr",
        "Save edited OCR",
        class = "btn-warning"
      )
    )
  })

}

shinyApp(ui, server)
# ---- End of app.R -------------------------------------------------
# Old version together
# shiny::runApp(host = "0.0.0.0", port = 8501)