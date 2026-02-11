# app.R --------------------------------------------------------------
# Local OCR + Field Extraction (Ollama + Shiny)
# - OCR: PDF/PNG/JPG via Ollama vision model
# - Extract: JSON fields via Ollama text model
# - Single + Bulk modes
# - Profiles + Exports

library(shiny)
library(bslib)
library(ollamar)
library(jsonlite)
library(pdftools)

# -------------------------------------------------------------------
# Profiles
# -------------------------------------------------------------------
profiles_dir <- "profiles"
if (!dir.exists(profiles_dir)) dir.create(profiles_dir, recursive = TRUE)

sanitize_name <- function(x) {
  x <- trimws(x)
  x <- gsub("[^A-Za-z0-9_-]", "_", x)
  if (!nzchar(x)) "" else x
}

profile_path <- function(name) {
  file.path(profiles_dir, paste0(sanitize_name(name), ".json"))
}

list_profiles <- function() {
  files <- list.files(profiles_dir, pattern = "\\.json$", full.names = FALSE)
  sub("\\.json$", "", files)
}

save_profile <- function(name, input) {
  nm <- sanitize_name(name)
  if (!nzchar(nm)) stop("Profile name is empty.")
  path <- profile_path(nm)

  prof <- list(
    field_list    = input$field_list,
    user_prompt   = input$user_prompt,
    ocr_model     = input$ocr_model,
    extract_model = input$extract_model,
    dpi           = input$dpi
  )
  write_json(prof, path, pretty = TRUE, auto_unbox = TRUE)
  nm
}

load_profile <- function(name) {
  nm <- sanitize_name(name)
  path <- profile_path(nm)
  if (!file.exists(path)) stop("Profile not found: ", nm)
  fromJSON(path, simplifyVector = TRUE)
}

delete_profile <- function(name) {
  nm <- sanitize_name(name)
  path <- profile_path(nm)
  if (!file.exists(path)) stop("Profile not found: ", nm)
  file.remove(path)
}

# -------------------------------------------------------------------
# OCR helpers
# -------------------------------------------------------------------
pdf_to_png <- function(pdf_path, dpi = 200L) {
  if (!file.exists(pdf_path)) stop("PDF not found: ", pdf_path)
  pngs <- pdftools::pdf_convert(pdf_path, dpi = dpi)
  if (!length(pngs)) stop("No pages converted from PDF: ", pdf_path)
  pngs
}

ocr_image_ollama <- function(image_path,
                             model = "mistral-small3.2",
                             prompt = "Transcribe all text as plain UTF-8 text, preserving reading order.",
                             temperature = 0) {
  if (!file.exists(image_path)) stop("Image not found: ", image_path)

  generate(
    model       = model,
    prompt      = prompt,
    images      = image_path,
    stream      = FALSE,
    output      = "text",
    temperature = temperature
  )
}

make_combined_text <- function(page_numbers, page_texts) {
  paste(
    sprintf("---- PAGE %d ----\n%s", page_numbers, page_texts),
    collapse = "\n\n"
  )
}

ocr_any_file <- function(file_path,
                         model = "mistral-small3.2",
                         dpi = 200L,
                         pages = NULL,
                         progress_cb = NULL,
                         prompt = "Transcribe all text as plain UTF-8 text, preserving reading order.") {
  ext <- tolower(tools::file_ext(file_path))

  if (ext == "pdf") {
    pngs <- pdf_to_png(file_path, dpi = dpi)
    n_total <- length(pngs)

    page_numbers <- seq_len(n_total)
    if (!is.null(pages)) {
      pages <- unique(as.integer(pages))
      pages <- pages[pages >= 1 & pages <= n_total]
      if (!length(pages)) stop("No valid pages after filtering page range.")
      pngs <- pngs[pages]
      page_numbers <- pages
    }
  } else if (ext %in% c("png", "jpg", "jpeg")) {
    pngs <- file_path
    page_numbers <- 1L
  } else {
    stop("Unsupported file type: ", ext)
  }

  n <- length(pngs)
  page_texts <- character(n)

  for (i in seq_len(n)) {
    if (!is.null(progress_cb)) progress_cb(i, n)
    page_texts[i] <- ocr_image_ollama(
      image_path  = pngs[i],
      model       = model,
      prompt      = prompt,
      temperature = 0
    )
  }

  list(
    page_paths    = pngs,
    page_numbers  = page_numbers,
    text_per_page = page_texts,
    combined_text = make_combined_text(page_numbers, page_texts)
  )
}

# -------------------------------------------------------------------
# Extraction helpers
# -------------------------------------------------------------------
extract_json_substring <- function(x) {
  start <- regexpr("\\{", x)
  end   <- regexpr(".*\\}", x)
  if (start[1] == -1 || end[1] == -1) return(x)
  substr(x, start[1], start[1] + attr(end, "match.length") - 1)
}

normalize_fields <- function(field_list_text) {
  fields <- strsplit(field_list_text, "\\r?\\n")[[1]]
  fields <- trimws(fields)
  fields[nzchar(fields)]
}

build_extraction_messages <- function(fields, user_prompt, ocr_text) {
  system_msg <- paste0(
    "You are an information extraction model. ",
    "Fill requested fields based on OCR text and provide confidence in [0,1]. ",
    "If missing, set value=\"\" and confidence=0.\n\n",
    "Return ONLY valid JSON:\n",
    "{\n",
    "  \"fields\": [\n",
    "    { \"name\": \"FieldName1\", \"value\": \"string\", \"confidence\": 0.0 }\n",
    "  ]\n",
    "}\n",
    "No markdown, no extra text."
  )

  user_msg <- paste0(
    "USER INSTRUCTIONS:\n", user_prompt, "\n\n",
    "FIELDS TO EXTRACT:\n", paste0("- ", fields, collapse = "\n"), "\n\n",
    "OCR TEXT:\n", ocr_text
  )

  list(system = system_msg, user = user_msg)
}

run_extraction <- function(model, fields, user_prompt, ocr_text) {
  # protect model context a bit
  max_chars <- 32000L
  if (nchar(ocr_text) > max_chars) ocr_text <- substr(ocr_text, 1, max_chars)

  msgs <- build_extraction_messages(fields, user_prompt, ocr_text)

  resp_text <- if (identical(trimws(model), "gpt-oss")) {
    # completion-style
    generate(
      model       = model,
      prompt      = paste(msgs$system, "\n\n---\n\n", msgs$user, sep = ""),
      stream      = FALSE,
      output      = "text",
      temperature = 0
    )
  } else {
    # chat-style
    chat(
      model = model,
      messages = list(
        list(role = "system", content = msgs$system),
        list(role = "user",   content = msgs$user)
      ),
      stream      = FALSE,
      output      = "text",
      temperature = 0
    )
  }

  if (is.null(resp_text) || !nzchar(resp_text)) stop("Extraction returned empty response.")

  json_str <- extract_json_substring(resp_text)
  parsed <- fromJSON(json_str, simplifyVector = TRUE)
  if (is.null(parsed$fields)) stop("Response JSON missing 'fields'.")

  df <- as.data.frame(parsed$fields, stringsAsFactors = FALSE)
  names(df) <- tolower(names(df))
  if (!"name" %in% names(df)) df$name <- NA_character_
  if (!"value" %in% names(df)) df$value <- ""
  if (!"confidence" %in% names(df)) df$confidence <- 0

  list(df = df, raw = resp_text)
}

# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
light_theme <- bs_theme(version = 5, bootswatch = "flatly")
dark_theme  <- bs_theme(version = 5, bootswatch = "darkly")

ui <- fluidPage(
  theme = light_theme,
  tags$head(tags$style(HTML("
    #original_page_image img, #bulk_page_image img { max-width: 100%; height: auto; display: block; }
    .ocr-text-wrap { max-height: 80vh; overflow-y: auto; padding: .5rem; border-radius: .5rem;
                     border: 1px solid rgba(0,0,0,.12); background: rgba(0,0,0,.02); }
    textarea { width: 100% !important; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
  "))),
  titlePanel("Document Dissector: Local OCR + Field Extraction (Ollama + Shiny)"),

  sidebarLayout(
    sidebarPanel(
      h4("Appearance"),
      checkboxInput("dark_mode", "Dark mode", value = FALSE),
      hr(),

      h4("Single file"),
      fileInput("file", "PDF / PNG / JPEG", accept = c(".pdf", ".png", ".jpg", ".jpeg")),
      sliderInput("dpi", "PDF render DPI", min = 100, max = 300, value = 200, step = 50),
      fluidRow(
        column(6, numericInput("page_start", "First page", value = NA, min = 1, step = 1)),
        column(6, numericInput("page_end",   "Last page",  value = NA, min = 1, step = 1))
      ),
      textInput("ocr_model", "OCR model", value = "mistral-small3.2"),
      actionButton("run_ocr", "Run OCR", class = "btn-primary"),
      br(), br(),

      h4("Extraction"),
      textAreaInput("field_list", "Fields (one per line)", "Field 1\nField 2\nField 3", rows = 5),
      textAreaInput("user_prompt", "Extra instructions", "Extract the requested fields as accurately as possible.", rows = 3),
      textInput("extract_model", "Extraction model", value = "gpt-oss"),
      fileInput("extract_text_file", "Optional: OCR text file (.txt)", accept = c(".txt", ".log", ".text")),
      actionButton("run_extract", "Run Extraction", class = "btn-success"),
      hr(),

      h4("Profiles"),
      textInput("profile_name", "Profile name", value = ""),
      fluidRow(
        column(6, actionButton("save_profile", "Save")),
        column(6, actionButton("delete_profile", "Delete", class = "btn-danger"))
      ),
      selectInput("load_profile_name", "Load profile", choices = list_profiles()),
      actionButton("load_profile", "Load"),
      hr(),

      h4("Exports (single)"),
      downloadButton("download_csv", "CSV"),
      downloadButton("download_json", "JSON"),
      downloadButton("download_ocr_text", "OCR text"),
      hr(),

      h4("Bulk"),
      fileInput("bulk_files", "Bulk docs", multiple = TRUE, accept = c(".pdf", ".png", ".jpg", ".jpeg")),
      div(
        class = "d-flex gap-2",
        actionButton("run_bulk_ocr", "1) Bulk OCR", class = "btn-warning flex-fill"),
        actionButton("run_bulk_extract", "2) Bulk Extract", class = "btn-danger flex-fill")
      ),
      br(), br(),
      downloadButton("download_bulk_csv", "Bulk CSV"),
      downloadButton("download_bulk_json", "Bulk JSON")
    ),

    mainPanel(
      tabsetPanel(
        id = "mode_tabs",

        tabPanel(
          "Single",
          uiOutput("page_selector_ui"),
          fluidRow(
            column(6, imageOutput("original_page_image")),
            column(
              6,
              div(
                class = "ocr-text-wrap",
                tags$strong("OCR text (editable, per page)"),
                textAreaInput("edit_page_text", label = NULL, value = "",
                  rows  = 24,
                  resize = "vertical",
                  width  = "100%"
                ),
                div(
                  class = "d-flex gap-2",
                  actionButton("save_page_edit", "Save edit", class = "btn-warning flex-fill"),
                  actionButton("reset_page_edit", "Reset", class = "btn-secondary flex-fill")
                )
              )
            )
          ),
          hr(),
          tabsetPanel(
            id = "tabs",
            tabPanel("OCR Text", verbatimTextOutput("ocr_output")),
            tabPanel("Extracted Fields", tableOutput("extract_table")),
            tabPanel("Raw JSON", verbatimTextOutput("raw_json_output"))
          )
        ),

        tabPanel(
          "Bulk",
          fluidRow(
            column(6, selectInput("bulk_file_select", "File", choices = character(0))),
            column(6, uiOutput("bulk_page_selector_ui"))
          ),
          fluidRow(
            column(6, imageOutput("bulk_page_image")),
            column(
              6,
              div(
                class = "ocr-text-wrap",
                tags$strong("Bulk OCR text (editable, per page)"),
                textAreaInput("bulk_edit_page_text", label = NULL, value = "",
                  rows  = 24,
                  resize = "vertical",
                  width  = "100%"
                ),
                div(
                  class = "d-flex gap-2",
                  actionButton("bulk_save_page_edit", "Save edit", class = "btn-warning flex-fill"),
                  actionButton("bulk_reset_page_edit", "Reset", class = "btn-secondary flex-fill")
                )
              )
            )
          ),
          hr(),
          h4("Aggregated results"),
          tableOutput("bulk_table")
        )
      )
    )
  )
)

# -------------------------------------------------------------------
# Server
# -------------------------------------------------------------------
server <- function(input, output, session) {

  # Themes
  observeEvent(input$dark_mode, {
    session$setCurrentTheme(if (isTRUE(input$dark_mode)) dark_theme else light_theme)
  }, ignoreInit = TRUE)

  # Single state
  single_ocr    <- reactiveVal(NULL)  # list(page_paths, page_numbers, text_per_page, combined_text)
  single_edits  <- reactiveVal(NULL)  # character vector per page
  single_extract_df  <- reactiveVal(NULL)
  single_extract_raw <- reactiveVal(NULL)

  # Bulk state
  bulk_ocr     <- reactiveVal(list())  # fname -> ocr list
  bulk_edits   <- reactiveVal(list())  # fname -> char vec
  bulk_extract <- reactiveVal(list())  # fname -> list(df, raw)

  # ---------------------------
  # Single: OCR
  # ---------------------------
  output$page_selector_ui <- renderUI({
    res <- single_ocr()
    if (is.null(res) || is.null(res$text_per_page)) return(helpText("Run OCR to enable preview."))
    n <- length(res$text_per_page)
    if (n <= 1) return(tags$strong("Single-page document"))
    selectInput("page_index", "Page", choices = seq_len(n), selected = 1)
  })

  observeEvent(list(input$page_index, single_ocr(), single_edits()), {
    res <- single_ocr()
    if (is.null(res) || is.null(res$text_per_page)) {
      updateTextAreaInput(session, "edit_page_text", value = "")
      return()
    }
    n <- length(res$text_per_page)
    idx <- if (!is.null(input$page_index) && !is.na(input$page_index)) {
      max(1L, min(n, as.integer(input$page_index)))
    } else 1L

    edt <- single_edits()
    page_text <- if (!is.null(edt) && length(edt) >= idx) edt[[idx]] else res$text_per_page[[idx]]
    updateTextAreaInput(session, "edit_page_text", value = page_text)
  }, ignoreInit = FALSE)

  output$original_page_image <- renderImage({
    res <- single_ocr()
    if (is.null(res) || is.null(res$page_paths)) {
      return(list(src = "", contentType = NULL, alt = "No preview yet."))
    }

    n <- length(res$page_paths)
    idx <- if (!is.null(input$page_index) && !is.na(input$page_index)) {
      max(1L, min(n, as.integer(input$page_index)))
    } else 1L

    img <- res$page_paths[idx]
    ext <- tolower(tools::file_ext(img))
    ctype <- if (ext %in% c("jpg", "jpeg")) "image/jpeg" else "image/png"
    list(src = img, contentType = ctype, alt = paste("Page", idx))
  }, deleteFile = FALSE)

  observeEvent(input$run_ocr, {
    req(input$file)
    single_ocr(NULL); single_edits(NULL)
    single_extract_df(NULL); single_extract_raw(NULL)

    file_path <- input$file$datapath
    ext <- tolower(tools::file_ext(file_path))
    pages_to_ocr <- NULL

    if (ext == "pdf") {
      info <- pdftools::pdf_info(file_path)
      n_total <- info$pages

      p_start <- if (is.null(input$page_start) || is.na(input$page_start)) 1L else max(1L, as.integer(input$page_start))
      p_end   <- if (is.null(input$page_end)   || is.na(input$page_end))   n_total else min(n_total, as.integer(input$page_end))

      if (p_start <= p_end) pages_to_ocr <- seq(p_start, p_end) else pages_to_ocr <- seq_len(n_total)
    }

    withProgress(message = "Running OCR...", value = 0, {
      progress_cb <- function(i, n) incProgress(1 / n, detail = sprintf("Page %d of %d", i, n))
      res <- tryCatch(
        ocr_any_file(
          file_path   = file_path,
          model       = input$ocr_model,
          dpi         = input$dpi,
          pages       = pages_to_ocr,
          progress_cb = progress_cb
        ),
        error = function(e) {
          showNotification(paste("OCR error:", e$message), type = "error")
          NULL
        }
      )
      single_ocr(res)
      if (!is.null(res) && !is.null(res$text_per_page)) single_edits(res$text_per_page)
    })
  })

  observeEvent(input$save_page_edit, {
    res <- single_ocr()
    req(res, res$text_per_page)

    n <- length(res$text_per_page)
    idx <- if (!is.null(input$page_index) && !is.na(input$page_index)) {
      max(1L, min(n, as.integer(input$page_index)))
    } else 1L

    edt <- single_edits()
    if (is.null(edt) || length(edt) != n) edt <- res$text_per_page
    edt[[idx]] <- if (is.null(input$edit_page_text)) "" else input$edit_page_text
    single_edits(edt)
    showNotification(paste("Saved edits for page", idx), type = "message")
  })

  observeEvent(input$reset_page_edit, {
    res <- single_ocr()
    req(res, res$text_per_page)

    n <- length(res$text_per_page)
    idx <- if (!is.null(input$page_index) && !is.na(input$page_index)) {
      max(1L, min(n, as.integer(input$page_index)))
    } else 1L

    edt <- single_edits()
    if (is.null(edt) || length(edt) != n) edt <- res$text_per_page
    edt[[idx]] <- res$text_per_page[[idx]]
    single_edits(edt)
    updateTextAreaInput(session, "edit_page_text", value = res$text_per_page[[idx]])
    showNotification(paste("Reset page", idx), type = "message")
  })

  output$ocr_output <- renderText({
    res <- single_ocr()
    if (is.null(res)) return("Upload a file and click 'Run OCR'.")
    edt <- single_edits()
    if (!is.null(edt) && length(edt) == length(res$page_numbers)) {
      make_combined_text(res$page_numbers, edt)
    } else {
      res$combined_text
    }
  })

  # ---------------------------
  # Single: Extraction
  # ---------------------------
  observeEvent(input$run_extract, {
    fields <- normalize_fields(input$field_list)
    if (!length(fields)) {
      showNotification("Provide at least one field.", type = "warning")
      return()
    }

    uploaded_text <- NULL
    if (!is.null(input$extract_text_file) && nrow(input$extract_text_file) > 0) {
      uploaded_text <- paste(readLines(input$extract_text_file$datapath, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
    }

    res <- single_ocr()
    if (is.null(uploaded_text) || !nzchar(uploaded_text)) req(res)

    ocr_text <- if (!is.null(uploaded_text) && nzchar(uploaded_text)) {
      uploaded_text
    } else {
      edt <- single_edits()
      if (!is.null(edt) && length(edt) == length(res$page_numbers)) {
        make_combined_text(res$page_numbers, edt)
      } else {
        res$combined_text
      }
    }

    withProgress(message = "Running extraction...", value = 0, {
      incProgress(0.2, detail = "Calling model...")
      out <- tryCatch(
        run_extraction(input$extract_model, fields, input$user_prompt, ocr_text),
        error = function(e) {
          showNotification(paste("Extraction error:", e$message), type = "error")
          NULL
        }
      )
      if (!is.null(out)) {
        single_extract_df(out$df)
        single_extract_raw(out$raw)
        updateTabsetPanel(session, "tabs", selected = "Extracted Fields")
      }
    })
  })

  output$extract_table <- renderTable(single_extract_df(), bordered = TRUE, striped = TRUE, na = "")
  output$raw_json_output <- renderText({
    x <- single_extract_raw()
    if (is.null(x)) "No extraction yet." else x
  })

  # ---------------------------
  # Profiles
  # ---------------------------
  observe({
    updateSelectInput(session, "load_profile_name", choices = list_profiles())
  })

  observeEvent(input$save_profile, {
    if (!nzchar(trimws(input$profile_name))) {
      showNotification("Enter a profile name.", type = "warning")
      return()
    }
    tryCatch({
      nm <- save_profile(input$profile_name, input)
      showNotification(paste("Saved profile:", nm), type = "message")
      updateSelectInput(session, "load_profile_name", choices = list_profiles(), selected = nm)
    }, error = function(e) showNotification(paste("Save failed:", e$message), type = "error"))
  })

  observeEvent(input$load_profile, {
    nm <- input$load_profile_name
    if (!nzchar(nm)) return()
    prof <- tryCatch(load_profile(nm), error = function(e) {
      showNotification(paste("Load failed:", e$message), type = "error"); NULL
    })
    if (is.null(prof)) return()

    if (!is.null(prof$field_list))    updateTextAreaInput(session, "field_list", value = prof$field_list)
    if (!is.null(prof$user_prompt))   updateTextAreaInput(session, "user_prompt", value = prof$user_prompt)
    if (!is.null(prof$ocr_model))     updateTextInput(session, "ocr_model", value = prof$ocr_model)
    if (!is.null(prof$extract_model)) updateTextInput(session, "extract_model", value = prof$extract_model)
    if (!is.null(prof$dpi))           updateSliderInput(session, "dpi", value = prof$dpi)

    showNotification(paste("Loaded profile:", nm), type = "message")
  })

  observeEvent(input$delete_profile, {
    nm <- input$load_profile_name
    if (!nzchar(nm)) return()
    tryCatch({
      delete_profile(nm)
      showNotification(paste("Deleted profile:", nm), type = "message")
      updateSelectInput(session, "load_profile_name", choices = list_profiles())
    }, error = function(e) showNotification(paste("Delete failed:", e$message), type = "error"))
  })

  # ---------------------------
  # Single downloads
  # ---------------------------
  output$download_csv <- downloadHandler(
    filename = function() "fields.csv",
    content = function(file) {
      df <- single_extract_df()
      if (is.null(df)) df <- data.frame()
      write.csv(df, file, row.names = FALSE)
    }
  )

  output$download_json <- downloadHandler(
    filename = function() "fields.json",
    content = function(file) {
      df <- single_extract_df()
      if (is.null(df) || !nrow(df)) {
        writeLines("{}", file)
        return()
      }
      out <- setNames(
        lapply(seq_len(nrow(df)), function(i) list(value = df$value[i], confidence = df$confidence[i])),
        df$name
      )
      write_json(out, file, pretty = TRUE, auto_unbox = TRUE)
    }
  )

  output$download_ocr_text <- downloadHandler(
    filename = function() "ocr.txt",
    content = function(file) {
      res <- single_ocr()
      if (is.null(res)) {
        writeLines("No OCR result available.", file)
        return()
      }
      edt <- single_edits()
      txt <- if (!is.null(edt) && length(edt) == length(res$page_numbers)) {
        make_combined_text(res$page_numbers, edt)
      } else {
        res$combined_text
      }
      writeLines(txt, file, useBytes = TRUE)
    }
  )

  # ---------------------------
  # Bulk: OCR + edits
  # ---------------------------
  observeEvent(input$run_bulk_ocr, {
    req(input$bulk_files)
    bulk_ocr(list()); bulk_edits(list()); bulk_extract(list())

    files <- input$bulk_files
    n_files <- nrow(files)

    withProgress(message = "Running bulk OCR...", value = 0, {
      ocr_map <- list()
      edit_map <- list()

      for (i in seq_len(n_files)) {
        incProgress(1 / n_files, detail = sprintf("File %d/%d: %s", i, n_files, files$name[i]))
        fname <- files$name[i]
        fp <- files$datapath[i]

        res <- tryCatch(
          ocr_any_file(fp, model = input$ocr_model, dpi = input$dpi),
          error = function(e) {
            showNotification(paste("Bulk OCR error:", fname, "-", e$message), type = "error")
            NULL
          }
        )
        if (is.null(res)) next
        ocr_map[[fname]] <- res
        edit_map[[fname]] <- res$text_per_page
      }

      bulk_ocr(ocr_map)
      bulk_edits(edit_map)

      if (length(ocr_map)) {
        updateSelectInput(session, "bulk_file_select", choices = names(ocr_map), selected = names(ocr_map)[1])
      }
    })
  })

  output$bulk_page_selector_ui <- renderUI({
    ocr_map <- bulk_ocr()
    fname <- input$bulk_file_select
    if (!nzchar(fname) || is.null(ocr_map[[fname]])) return(helpText("Run bulk OCR, then select a file."))

    n <- length(ocr_map[[fname]]$text_per_page)
    if (n <= 1) return(tags$strong("Single-page document"))
    selectInput("bulk_page_index", "Page", choices = seq_len(n), selected = 1)
  })

  observeEvent(list(input$bulk_file_select, input$bulk_page_index, bulk_ocr(), bulk_edits()), {
    ocr_map <- bulk_ocr()
    edit_map <- bulk_edits()
    fname <- input$bulk_file_select

    if (!nzchar(fname) || is.null(ocr_map[[fname]])) {
      updateTextAreaInput(session, "bulk_edit_page_text", value = "")
      return()
    }

    res <- ocr_map[[fname]]
    n <- length(res$text_per_page)
    idx <- if (!is.null(input$bulk_page_index) && !is.na(input$bulk_page_index)) {
      max(1L, min(n, as.integer(input$bulk_page_index)))
    } else 1L

    vec <- edit_map[[fname]]
    if (is.null(vec) || length(vec) != n) vec <- res$text_per_page
    updateTextAreaInput(session, "bulk_edit_page_text", value = vec[[idx]])
  }, ignoreInit = TRUE)

  output$bulk_page_image <- renderImage({
    ocr_map <- bulk_ocr()
    fname <- input$bulk_file_select
    if (!nzchar(fname) || is.null(ocr_map[[fname]])) return(list(src = "", contentType = NULL, alt = "No preview yet."))

    res <- ocr_map[[fname]]
    n <- length(res$page_paths)
    idx <- if (!is.null(input$bulk_page_index) && !is.na(input$bulk_page_index)) {
      max(1L, min(n, as.integer(input$bulk_page_index)))
    } else 1L

    img <- res$page_paths[idx]
    ext <- tolower(tools::file_ext(img))
    ctype <- if (ext %in% c("jpg", "jpeg")) "image/jpeg" else "image/png"
    list(src = img, contentType = ctype, alt = paste(fname, "- page", idx))
  }, deleteFile = FALSE)

  observeEvent(input$bulk_save_page_edit, {
    ocr_map <- bulk_ocr()
    edit_map <- bulk_edits()
    fname <- input$bulk_file_select
    if (!nzchar(fname) || is.null(ocr_map[[fname]])) return()

    res <- ocr_map[[fname]]
    n <- length(res$text_per_page)
    idx <- if (!is.null(input$bulk_page_index) && !is.na(input$bulk_page_index)) {
      max(1L, min(n, as.integer(input$bulk_page_index)))
    } else 1L

    vec <- edit_map[[fname]]
    if (is.null(vec) || length(vec) != n) vec <- res$text_per_page
    vec[[idx]] <- if (is.null(input$bulk_edit_page_text)) "" else input$bulk_edit_page_text
    edit_map[[fname]] <- vec
    bulk_edits(edit_map)

    showNotification(paste("Saved edits:", fname, "page", idx), type = "message")
  })

  observeEvent(input$bulk_reset_page_edit, {
    ocr_map <- bulk_ocr()
    edit_map <- bulk_edits()
    fname <- input$bulk_file_select
    if (!nzchar(fname) || is.null(ocr_map[[fname]])) return()

    res <- ocr_map[[fname]]
    n <- length(res$text_per_page)
    idx <- if (!is.null(input$bulk_page_index) && !is.na(input$bulk_page_index)) {
      max(1L, min(n, as.integer(input$bulk_page_index)))
    } else 1L

    vec <- edit_map[[fname]]
    if (is.null(vec) || length(vec) != n) vec <- res$text_per_page
    vec[[idx]] <- res$text_per_page[[idx]]
    edit_map[[fname]] <- vec
    bulk_edits(edit_map)

    updateTextAreaInput(session, "bulk_edit_page_text", value = res$text_per_page[[idx]])
    showNotification(paste("Reset:", fname, "page", idx), type = "message")
  })

  # ---------------------------
  # Bulk: extraction + aggregation + downloads
  # ---------------------------
  observeEvent(input$run_bulk_extract, {
    ocr_map <- bulk_ocr()
    req(length(ocr_map) > 0)

    fields <- normalize_fields(input$field_list)
    if (!length(fields)) {
      showNotification("Provide at least one field.", type = "warning")
      return()
    }

    withProgress(message = "Running bulk extraction...", value = 0, {
      out_map <- list()
      fns <- names(ocr_map)
      n <- length(fns)

      for (i in seq_len(n)) {
        fname <- fns[i]
        incProgress(1 / n, detail = sprintf("File %d/%d: %s", i, n, fname))

        res <- ocr_map[[fname]]
        vec <- bulk_edits()[[fname]]
        if (is.null(vec) || length(vec) != length(res$page_numbers)) vec <- res$text_per_page

        combined <- make_combined_text(res$page_numbers, vec)

        out <- tryCatch(
          run_extraction(input$extract_model, fields, input$user_prompt, combined),
          error = function(e) {
            showNotification(paste("Bulk extraction error:", fname, "-", e$message), type = "error")
            NULL
          }
        )
        if (!is.null(out)) out_map[[fname]] <- out
      }

      bulk_extract(out_map)
    })
  })

  bulk_table <- reactive({
    ext <- bulk_extract()
    if (!length(ext)) return(NULL)

    all_fields <- sort(unique(unlist(lapply(ext, function(x) x$df$name))))
    tab <- data.frame(Field = all_fields, stringsAsFactors = FALSE)

    for (fname in names(ext)) {
      df <- ext[[fname]]$df
      tab[[fname]] <- df$value[match(all_fields, df$name)]
    }
    tab
  })

  output$bulk_table <- renderTable(bulk_table(), bordered = TRUE, striped = TRUE, na = "")

  output$download_bulk_csv <- downloadHandler(
    filename = function() paste0("bulk_results_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv"),
    content = function(file) {
      tab <- bulk_table()
      if (is.null(tab)) tab <- data.frame()
      write.csv(tab, file, row.names = FALSE)
    }
  )

  output$download_bulk_json <- downloadHandler(
    filename = function() paste0("bulk_results_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".json"),
    content = function(file) {
      tab <- bulk_table()
      if (is.null(tab) || !nrow(tab)) {
        writeLines("{}", file)
        return()
      }
      fields <- tab$Field
      files <- setdiff(names(tab), "Field")

      out <- setNames(
        lapply(seq_along(fields), function(i) {
          vals <- as.list(tab[i, files, drop = FALSE])
          lapply(vals, function(x) if (is.na(x)) "" else x)
        }),
        fields
      )
      write_json(out, file, pretty = TRUE, auto_unbox = TRUE)
    }
  )
}

shinyApp(ui, server)
# -------------------------------------------------------------- end
