# app.R --------------------------------------------------------------
library(shiny)
library(bslib)
library(ollamar)
library(jsonlite)
library(pdftools)

# ---- Profiles -------------------------------------------------------
profiles_dir <- "profiles"
dir.create(profiles_dir, showWarnings = FALSE, recursive = TRUE)

clean_name <- function(x) {
  x <- trimws(x)
  gsub("[^A-Za-z0-9_-]", "_", x)
}

list_profiles <- function() {
  sub("\\.json$", "", list.files(profiles_dir, pattern = "\\.json$", full.names = FALSE))
}

save_profile <- function(name, input) {
  name <- clean_name(name)
  stopifnot(nzchar(name))
  prof <- list(
    field_list    = input$field_list,
    user_prompt   = input$user_prompt,
    ocr_model     = input$ocr_model,
    extract_model = input$extract_model,
    dpi           = input$dpi
  )
  write_json(prof, file.path(profiles_dir, paste0(name, ".json")),
             pretty = TRUE, auto_unbox = TRUE)
  name
}

load_profile <- function(name) {
  name <- clean_name(name)
  path <- file.path(profiles_dir, paste0(name, ".json"))
  if (!file.exists(path)) stop("Profile not found: ", name)
  fromJSON(path, simplifyVector = TRUE)
}

delete_profile <- function(name) {
  name <- clean_name(name)
  path <- file.path(profiles_dir, paste0(name, ".json"))
  if (!file.exists(path)) stop("Profile not found: ", name)
  file.remove(path)
}

# ---- Session image cache (save + cleanup) ---------------------------
ocr_images_base_dir <- "ocr_images"
dir.create(ocr_images_base_dir, showWarnings = FALSE, recursive = TRUE)

sanitize_prefix <- function(x) {
  x <- tools::file_path_sans_ext(basename(x))
  x <- gsub("[^A-Za-z0-9_-]", "_", x)
  if (!nzchar(x)) "doc" else x
}

make_session_dir <- function(token) {
  d <- file.path(ocr_images_base_dir, paste0("session_", token))
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
  d
}

cleanup_images_keep_folder <- function(dir_path) {
  if (!dir.exists(dir_path)) return(invisible(NULL))
  f <- list.files(dir_path, recursive = TRUE, full.names = TRUE, include.dirs = FALSE)
  if (length(f)) unlink(f, force = TRUE)
  invisible(NULL)
}

# ---- OCR helpers ----------------------------------------------------
ocr_page_image_ollama <- function(image_path, model, prompt, temperature = 0) {
  if (!file.exists(image_path)) stop("Image not found: ", image_path)
  generate(
    model = model,
    prompt = prompt,
    images = image_path,
    stream = FALSE,
    output = "text",
    temperature = temperature
  )
}

convert_to_images <- function(file_path, dpi, pages, out_dir, prefix) {
  ext <- tolower(tools::file_ext(file_path))
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  if (ext == "pdf") {
    info <- pdftools::pdf_info(file_path)
    n_total <- info$pages

    if (is.null(pages)) pages <- seq_len(n_total)
    pages <- unique(as.integer(pages))
    pages <- pages[!is.na(pages) & pages >= 1 & pages <= n_total]
    if (!length(pages)) stop("No valid pages to OCR.")

    filenames <- file.path(out_dir, sprintf("%s_p%03d.png", prefix, pages))
    paths <- pdftools::pdf_convert(pdf = file_path, dpi = dpi, pages = pages, filenames = filenames)
    list(paths = paths, page_numbers = pages)

  } else if (ext %in% c("png", "jpg", "jpeg")) {
    out_ext <- if (ext == "png") "png" else "jpg"
    dest <- file.path(out_dir, sprintf("%s_p001.%s", prefix, out_ext))
    if (!file.copy(file_path, dest, overwrite = TRUE)) stop("Failed to copy image.")
    list(paths = dest, page_numbers = 1L)

  } else {
    stop("Unsupported file type: ", ext)
  }
}

combine_pages <- function(page_numbers, page_texts) {
  paste(sprintf("---- PAGE %d ----\n%s", page_numbers, page_texts), collapse = "\n\n")
}

ocr_any_file_ollama <- function(file_path, model, dpi, prompt, pages, out_dir, prefix, progress_cb = NULL) {
  imgs <- convert_to_images(file_path, dpi, pages, out_dir, prefix)
  paths <- imgs$paths
  nums  <- imgs$page_numbers

  n <- length(paths)
  texts <- character(n)

  for (i in seq_len(n)) {
    if (!is.null(progress_cb)) progress_cb(i, n)
    texts[i] <- ocr_page_image_ollama(paths[i], model, prompt, temperature = 0)
  }

  list(
    text_per_page = texts,
    combined_text = combine_pages(nums, texts),
    page_paths    = paths,
    page_numbers  = nums
  )
}

# ---- Extraction helpers --------------------------------------------
extract_json_substring <- function(x) {
  s <- regexpr("\\{", x); e <- regexpr(".*\\}", x)
  if (s[1] == -1 || e[1] == -1) return(x)
  substr(x, s[1], s[1] + attr(e, "match.length") - 1)
}

SYSTEM_MSG <- paste0(
  "You are an information extraction model. The user will provide OCR text and a list of target fields. ",
  "Fill fields based on OCR text and provide confidence in [0,1]. If missing, value=\"\" and confidence=0.\n\n",
  "Return ONLY valid JSON:\n",
  "{\n",
  "  \"fields\": [\n",
  "    { \"name\": \"FieldName1\", \"value\": \"string\", \"confidence\": 0.0 }\n",
  "  ]\n",
  "}\n",
  "No markdown, no explanation, no extra text."
)

run_extraction <- function(model, fields, user_prompt, ocr_text_full) {
  fields <- trimws(fields); fields <- fields[nzchar(fields)]
  if (!length(fields)) stop("No fields provided.")

  max_chars <- 32000L
  ocr_text <- if (!is.null(ocr_text_full) && nchar(ocr_text_full) > max_chars)
    substr(ocr_text_full, 1, max_chars) else ocr_text_full

  user_msg <- paste0(
    "USER INSTRUCTIONS:\n", user_prompt, "\n\n",
    "FIELDS TO EXTRACT:\n- ", paste(fields, collapse = "\n- "), "\n\n",
    "OCR TEXT (possibly truncated):\n", ocr_text
  )

  resp <- if (identical(model, "gpt-oss")) {
    generate(model = model,
             prompt = paste(SYSTEM_MSG, "\n\n---\n\n", user_msg),
             stream = FALSE, output = "text", temperature = 0)
  } else {
    chat(model = model,
         messages = list(
           list(role = "system", content = SYSTEM_MSG),
           list(role = "user",   content = user_msg)
         ),
         stream = FALSE, output = "text", temperature = 0)
  }

  if (is.null(resp) || !nzchar(resp)) stop("Extraction returned empty response.")

  parsed <- fromJSON(extract_json_substring(resp), simplifyVector = TRUE)
  if (is.null(parsed$fields)) stop("Response JSON missing 'fields'.")

  df <- as.data.frame(parsed$fields, stringsAsFactors = FALSE)
  names(df) <- tolower(names(df))
  if (!"value" %in% names(df)) df$value <- ""
  if (!"confidence" %in% names(df)) df$confidence <- 0
  list(df = df[, c("name", "value", "confidence"), drop = FALSE], raw = resp)
}

spellcheck_pages <- function(pages) {
  if (!requireNamespace("hunspell", quietly = TRUE)) return(pages)
  lapply(pages, function(txt) {
    if (!nzchar(txt)) return(txt)
    bad <- hunspell::hunspell(txt)[[1]]
    for (w in bad) {
      sugg <- hunspell::hunspell_suggest(w)[[1]]
      if (length(sugg) == 1) {
        txt <- gsub(paste0("\\b", w, "\\b"), sugg[1], txt, perl = TRUE)
      }
    }
    txt
  }) |> unlist(use.names = FALSE)
}

# ---- UI ------------------------------------------------------------
light_theme <- bs_theme(version = 5, bootswatch = "flatly")
dark_theme  <- bs_theme(version = 5, bootswatch = "darkly")

ui <- fluidPage(
  theme = light_theme,
  tags$head(
    tags$style(HTML("
      .ocr-box {max-height: 80vh; overflow-y:auto; padding:.5rem; border-radius:.5rem; border:1px solid rgba(0,0,0,.1);}
      #original_page_image img,#bulk_page_image img{max-width:100%;height:auto;display:block;}
      textarea{font-family:monospace; white-space:pre-wrap; overflow-wrap:anywhere;}
    ")),
    tags$script(HTML("
      $(function() {
        if (window.matchMedia) {
          var mq = window.matchMedia('(prefers-color-scheme: dark)');
          Shiny.setInputValue('system_pref_dark', mq.matches, {priority: 'event'});
        }
      });
    "))
  ),

  titlePanel("Local OCR + Field Extraction (Ollama + R Shiny)"),

  sidebarLayout(
    sidebarPanel(
      checkboxInput("dark_mode", "Dark mode", FALSE),
      checkboxInput("auto_spellcheck", "Auto spellcheck edited pages before extraction", FALSE),
      tags$hr(),

      h4("Single"),
      fileInput("file", "PDF / PNG / JPEG", accept = c(".pdf", ".png", ".jpg", ".jpeg")),
      sliderInput("dpi", "PDF DPI", min = 100, max = 300, value = 300, step = 50),
      numericInput("page_start", "First page (blank=1)", value = NA, min = 1),
      numericInput("page_end",   "Last page (blank=all)", value = NA, min = 1),
      textInput("ocr_model", "OCR model", "mistral-small3.2"),
      actionButton("run_ocr", "Run OCR", class = "btn-primary"),
      uiOutput("ocr_spinner"),
      tags$hr(),

      h4("Extraction"),
      textAreaInput("field_list", "Fields (one per line)", "Field 1\nField 2\nField 3", rows = 5),
      textAreaInput("user_prompt", "Instructions", "Extract the requested fields as accurately as possible from the OCR text.", rows = 3),
      textInput("extract_model", "Extraction model", "gpt-oss"),
      fileInput("extract_text_file", "Optional OCR text (.txt)", accept = c(".txt", ".text", ".log")),
      actionButton("run_extract", "Run Extraction", class = "btn-success"),
      tags$hr(),

      h4("Profiles"),
      textInput("profile_name", "Name (for save)", ""),
      fluidRow(
        column(6, actionButton("save_profile", "Save")),
        column(6, actionButton("delete_profile", "Delete", class = "btn-danger"))
      ),
      selectInput("load_profile_name", "Load", choices = list_profiles()),
      actionButton("load_profile", "Load"),
      tags$hr(),

      h4("Downloads"),
      downloadButton("download_csv", "Single CSV"),
      downloadButton("download_json", "Single JSON"),
      downloadButton("download_ocr_text", "Single OCR .txt"),
      tags$hr(),

      h4("Bulk"),
      fileInput("bulk_files", "Bulk docs", multiple = TRUE, accept = c(".pdf", ".png", ".jpg", ".jpeg")),
      div(class="d-flex gap-2",
          actionButton("run_bulk_ocr", "1) Bulk OCR", class="btn-warning flex-fill"),
          actionButton("run_bulk_extract", "2) Bulk Extract", class="btn-danger flex-fill")
      ),
      br(),
      downloadButton("download_bulk_csv", "Bulk CSV"),
      downloadButton("download_bulk_json", "Bulk JSON")
    ),

    mainPanel(
      tabsetPanel(
        tabPanel("Single",
          uiOutput("page_selector_ui"),
          fluidRow(
            column(6, imageOutput("original_page_image")),
            column(6,
              div(class="ocr-box",
                h5("Page OCR (editable)"),
                textAreaInput("edit_page_text", NULL, "", rows = 22),
                div(class="d-flex gap-2",
                    actionButton("save_page_edit","Save", class="btn-warning flex-fill"),
                    actionButton("reset_page_edit","Reset", class="btn-secondary flex-fill")
                )
              )
            )
          ),
          tags$hr(),
          tabsetPanel(
            id="tabs",
            tabPanel("OCR Text", verbatimTextOutput("ocr_output")),
            tabPanel("Extracted Fields", tableOutput("extract_table")),
            tabPanel("Raw JSON", verbatimTextOutput("raw_json_output"))
          )
        ),

        tabPanel("Bulk",
          fluidRow(
            column(6, selectInput("bulk_file_select", "File", choices = character(0))),
            column(6, uiOutput("bulk_page_selector_ui"))
          ),
          fluidRow(
            column(6, imageOutput("bulk_page_image")),
            column(6,
              div(class="ocr-box",
                h5("Bulk page OCR (editable)"),
                textAreaInput("bulk_edit_page_text", NULL, "", rows = 22),
                div(class="d-flex gap-2",
                    actionButton("bulk_save_page_edit","Save", class="btn-warning flex-fill"),
                    actionButton("bulk_reset_page_edit","Reset", class="btn-secondary flex-fill")
                )
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

# ---- Server --------------------------------------------------------
server <- function(input, output, session) {
  # Per-session image folder + cleanup
  session_img_dir <- make_session_dir(session$token)
  session$onSessionEnded(function() cleanup_images_keep_folder(session_img_dir))

  # Theme toggle
  observeEvent(input$system_pref_dark, {
    if (isTRUE(input$system_pref_dark)) {
      session$setCurrentTheme(dark_theme)
      updateCheckboxInput(session, "dark_mode", value = TRUE)
    }
  }, once = TRUE)

  observeEvent(input$dark_mode, {
    session$setCurrentTheme(if (isTRUE(input$dark_mode)) dark_theme else light_theme)
  })

  # Reactive state
  ocr_running <- reactiveVal(FALSE)

  single_res   <- reactiveVal(NULL)
  single_edits <- reactiveVal(NULL)
  single_df    <- reactiveVal(NULL)
  single_raw   <- reactiveVal(NULL)

  bulk_res   <- reactiveVal(list())   # fname -> ocr list
  bulk_edits <- reactiveVal(list())   # fname -> character vector per page
  bulk_ext   <- reactiveVal(list())   # fname -> list(df, raw)

  # Helpers to get current page index safely
  page_idx <- function(n, idx_in) {
    idx <- suppressWarnings(as.integer(idx_in))
    if (is.na(idx) || idx < 1 || idx > n) 1L else idx
  }

  # Spinner
  output$ocr_spinner <- renderUI({
    if (!isTRUE(ocr_running())) return(NULL)
    tags$div(class="spinner-border text-secondary", role="status",
             tags$span(class="visually-hidden","Running OCR..."))
  })

  # ---- Single OCR --------------------------------------------------
  observeEvent(input$run_ocr, {
    req(input$file)
    ocr_running(TRUE)
    on.exit(ocr_running(FALSE), add = TRUE)

    single_res(NULL); single_edits(NULL); single_df(NULL); single_raw(NULL)

    file_path <- input$file$datapath
    ext <- tolower(tools::file_ext(file_path))
    dpi <- input$dpi
    model <- input$ocr_model
    prefix <- sanitize_prefix(input$file$name)

    pages <- NULL
    if (ext == "pdf") {
      n_total <- pdftools::pdf_info(file_path)$pages
      p1 <- if (is.na(input$page_start)) 1L else max(1L, as.integer(input$page_start))
      p2 <- if (is.na(input$page_end))   n_total else min(n_total, as.integer(input$page_end))
      if (p1 > p2) { p1 <- 1L; p2 <- n_total }
      pages <- seq(p1, p2)
    }

    withProgress("Running OCR...", value = 0, {
      cb <- function(i, n) incProgress(1/n, detail = sprintf("Page %d of %d", i, n))
      res <- tryCatch(
        ocr_any_file_ollama(
          file_path = file_path,
          model = model,
          dpi = dpi,
          prompt = "Transcribe all text in this scanned page as plain UTF-8 text, preserving reading order.",
          pages = pages,
          out_dir = session_img_dir,
          prefix = prefix,
          progress_cb = cb
        ),
        error = function(e) { showNotification(e$message, type="error"); NULL }
      )
      single_res(res)
      if (!is.null(res$text_per_page)) single_edits(res$text_per_page)
    })
  })

  # Single page selector UI
  output$page_selector_ui <- renderUI({
    res <- single_res()
    if (is.null(res$text_per_page)) return(helpText("Run OCR to enable preview."))
    n <- length(res$text_per_page)
    if (n <= 1) return(tags$strong("Single-page document"))
    selectInput("page_index", "Page", choices = seq_len(n), selected = 1)
  })

  # Update single edit box on page change
  observeEvent(list(input$page_index, single_res(), single_edits()), {
    res <- single_res(); edt <- single_edits()
    if (is.null(res$text_per_page)) {
      updateTextAreaInput(session, "edit_page_text", value = "")
      return()
    }
    n <- length(res$text_per_page)
    idx <- page_idx(n, input$page_index)
    txt <- if (!is.null(edt) && length(edt) >= idx) edt[[idx]] else res$text_per_page[[idx]]
    updateTextAreaInput(session, "edit_page_text", value = txt)
  }, ignoreInit = FALSE)

  observeEvent(input$save_page_edit, {
    res <- single_res(); req(res$text_per_page)
    n <- length(res$text_per_page)
    idx <- page_idx(n, input$page_index)
    edt <- single_edits(); if (is.null(edt) || length(edt) != n) edt <- res$text_per_page
    edt[[idx]] <- input$edit_page_text %||% ""
    single_edits(edt)
    showNotification(sprintf("Saved edits for page %d", idx), type="message")
  })

  observeEvent(input$reset_page_edit, {
    res <- single_res(); req(res$text_per_page)
    n <- length(res$text_per_page)
    idx <- page_idx(n, input$page_index)
    edt <- single_edits(); if (is.null(edt) || length(edt) != n) edt <- res$text_per_page
    edt[[idx]] <- res$text_per_page[[idx]]
    single_edits(edt)
    updateTextAreaInput(session, "edit_page_text", value = res$text_per_page[[idx]])
    showNotification(sprintf("Reset page %d", idx), type="message")
  })

  output$original_page_image <- renderImage({
    res <- single_res()
    if (is.null(res$page_paths)) return(list(src="", contentType=NULL, alt="No preview."))
    n <- length(res$page_paths)
    idx <- page_idx(n, input$page_index)
    p <- res$page_paths[[idx]]
    ext <- tolower(tools::file_ext(p))
    ctype <- if (ext == "png") "image/png" else "image/jpeg"
    list(src = p, contentType = ctype, alt = paste("Page", idx))
  }, deleteFile = FALSE)

  output$ocr_output <- renderText({
    res <- single_res()
    if (is.null(res)) return("Upload a file and click 'Run OCR'.")
    edt <- single_edits()
    if (!is.null(edt) && length(edt) == length(res$page_numbers)) {
      combine_pages(res$page_numbers, edt)
    } else {
      res$combined_text
    }
  })

  # ---- Extraction (single) ----------------------------------------
  observeEvent(input$run_extract, {
    # optional uploaded text
    uploaded_text <- NULL
    if (!is.null(input$extract_text_file) && nrow(input$extract_text_file) > 0) {
      uploaded_text <- paste(readLines(input$extract_text_file$datapath, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
    }

    res <- single_res()
    if (is.null(uploaded_text) || !nzchar(uploaded_text)) req(res)

    fields <- strsplit(input$field_list, "\\r?\\n")[[1]]
    fields <- trimws(fields); fields <- fields[nzchar(fields)]
    if (!length(fields)) { showNotification("Provide at least one field.", type="warning"); return() }

    ocr_text <- if (!is.null(uploaded_text) && nzchar(uploaded_text)) {
      uploaded_text
    } else {
      edt <- single_edits()
      if (!is.null(edt) && isTRUE(input$auto_spellcheck)) edt <- spellcheck_pages(edt)
      if (!is.null(edt) && length(edt) == length(res$page_numbers)) combine_pages(res$page_numbers, edt) else res$combined_text
    }

    withProgress("Running extraction...", value = 0, {
      incProgress(0.2)
      out <- tryCatch(
        run_extraction(input$extract_model, fields, input$user_prompt, ocr_text),
        error = function(e) { showNotification(e$message, type="error"); NULL }
      )
      if (!is.null(out)) {
        single_df(out$df)
        single_raw(out$raw)
        updateTabsetPanel(session, "tabs", selected = "Extracted Fields")
      }
    })
  })

  output$extract_table <- renderTable(single_df())
  output$raw_json_output <- renderText(single_raw() %||% "No extraction run yet.")

  # ---- Profiles UI -------------------------------------------------
  observe({ updateSelectInput(session, "load_profile_name", choices = list_profiles()) })

  observeEvent(input$save_profile, {
    if (!nzchar(trimws(input$profile_name))) { showNotification("Enter a profile name.", type="warning"); return() }
    tryCatch({
      nm <- save_profile(input$profile_name, input)
      updateSelectInput(session, "load_profile_name", choices = list_profiles(), selected = nm)
      showNotification(paste("Saved:", nm), type="message")
    }, error = function(e) showNotification(e$message, type="error"))
  })

  observeEvent(input$delete_profile, {
    nm <- input$load_profile_name
    if (!nzchar(nm)) return()
    tryCatch({
      delete_profile(nm)
      updateSelectInput(session, "load_profile_name", choices = list_profiles())
      showNotification(paste("Deleted:", nm), type="message")
    }, error = function(e) showNotification(e$message, type="error"))
  })

  observeEvent(input$load_profile, {
    nm <- input$load_profile_name
    if (!nzchar(nm)) return()
    prof <- tryCatch(load_profile(nm), error = function(e) { showNotification(e$message, type="error"); NULL })
    if (is.null(prof)) return()

    if (!is.null(prof$field_list))    updateTextAreaInput(session, "field_list", value = prof$field_list)
    if (!is.null(prof$user_prompt))   updateTextAreaInput(session, "user_prompt", value = prof$user_prompt)
    if (!is.null(prof$ocr_model))     updateTextInput(session, "ocr_model", value = prof$ocr_model)
    if (!is.null(prof$extract_model)) updateTextInput(session, "extract_model", value = prof$extract_model)
    if (!is.null(prof$dpi))           updateSliderInput(session, "dpi", value = prof$dpi)

    showNotification(paste("Loaded:", nm), type="message")
  })

  # ---- Bulk OCR ----------------------------------------------------
  observeEvent(input$run_bulk_ocr, {
    req(input$bulk_files)
    bulk_res(list()); bulk_edits(list()); bulk_ext(list())

    files <- input$bulk_files
    withProgress("Bulk OCR...", value = 0, {
      n <- nrow(files)
      ocr_map <- list()
      edt_map <- list()

      for (i in seq_len(n)) {
        incProgress(1/n, detail = sprintf("%d/%d: %s", i, n, files$name[i]))
        fname <- files$name[i]
        prefix <- sanitize_prefix(fname)

        res <- tryCatch(
          ocr_any_file_ollama(
            file_path = files$datapath[i],
            model = input$ocr_model,
            dpi = input$dpi,
            prompt = "Transcribe all text in this scanned page as plain UTF-8 text, preserving reading order.",
            pages = NULL,
            out_dir = session_img_dir,
            prefix = prefix
          ),
          error = function(e) { showNotification(paste(fname, ":", e$message), type="error"); NULL }
        )
        if (is.null(res)) next
        ocr_map[[fname]] <- res
        edt_map[[fname]] <- res$text_per_page
      }

      bulk_res(ocr_map)
      bulk_edits(edt_map)

      if (length(ocr_map)) {
        updateSelectInput(session, "bulk_file_select", choices = names(ocr_map), selected = names(ocr_map)[1])
      }
    })
  })

  # Bulk page selector
  output$bulk_page_selector_ui <- renderUI({
    o <- bulk_res(); fname <- input$bulk_file_select
    if (!nzchar(fname) || is.null(o[[fname]]$text_per_page)) return(helpText("Run bulk OCR first."))
    n <- length(o[[fname]]$text_per_page)
    if (n <= 1) return(tags$strong("Single-page document"))
    selectInput("bulk_page_index", "Page", choices = seq_len(n), selected = 1)
  })

  # Update bulk edit box on file/page change
  observeEvent(list(input$bulk_file_select, input$bulk_page_index, bulk_res(), bulk_edits()), {
    o <- bulk_res(); e <- bulk_edits(); fname <- input$bulk_file_select
    if (!nzchar(fname) || is.null(o[[fname]]$text_per_page)) {
      updateTextAreaInput(session, "bulk_edit_page_text", value = "")
      return()
    }
    n <- length(o[[fname]]$text_per_page)
    idx <- page_idx(n, input$bulk_page_index)
    vec <- e[[fname]]; if (is.null(vec) || length(vec) != n) vec <- o[[fname]]$text_per_page
    updateTextAreaInput(session, "bulk_edit_page_text", value = vec[[idx]])
  }, ignoreInit = TRUE)

  observeEvent(input$bulk_save_page_edit, {
    o <- bulk_res(); e <- bulk_edits(); fname <- input$bulk_file_select
    req(nzchar(fname), o[[fname]]$text_per_page)
    n <- length(o[[fname]]$text_per_page)
    idx <- page_idx(n, input$bulk_page_index)

    vec <- e[[fname]]; if (is.null(vec) || length(vec) != n) vec <- o[[fname]]$text_per_page
    vec[[idx]] <- input$bulk_edit_page_text %||% ""
    e[[fname]] <- vec
    bulk_edits(e)
    showNotification(sprintf("Saved %s - page %d", fname, idx), type="message")
  })

  observeEvent(input$bulk_reset_page_edit, {
    o <- bulk_res(); e <- bulk_edits(); fname <- input$bulk_file_select
    req(nzchar(fname), o[[fname]]$text_per_page)
    n <- length(o[[fname]]$text_per_page)
    idx <- page_idx(n, input$bulk_page_index)

    vec <- e[[fname]]; if (is.null(vec) || length(vec) != n) vec <- o[[fname]]$text_per_page
    vec[[idx]] <- o[[fname]]$text_per_page[[idx]]
    e[[fname]] <- vec
    bulk_edits(e)
    updateTextAreaInput(session, "bulk_edit_page_text", value = vec[[idx]])
    showNotification(sprintf("Reset %s - page %d", fname, idx), type="message")
  })

  output$bulk_page_image <- renderImage({
    o <- bulk_res(); fname <- input$bulk_file_select
    if (!nzchar(fname) || is.null(o[[fname]]$page_paths)) return(list(src="", contentType=NULL, alt="No preview."))
    n <- length(o[[fname]]$page_paths)
    idx <- page_idx(n, input$bulk_page_index)
    p <- o[[fname]]$page_paths[[idx]]
    ext <- tolower(tools::file_ext(p))
    ctype <- if (ext == "png") "image/png" else "image/jpeg"
    list(src = p, contentType = ctype, alt = paste(fname, "page", idx))
  }, deleteFile = FALSE)

  # ---- Bulk Extraction --------------------------------------------
  observeEvent(input$run_bulk_extract, {
    o <- bulk_res(); req(length(o))
    e <- bulk_edits()
    fields <- strsplit(input$field_list, "\\r?\\n")[[1]]
    fields <- trimws(fields); fields <- fields[nzchar(fields)]
    if (!length(fields)) { showNotification("Provide at least one field.", type="warning"); return() }

    out_map <- list()
    fns <- names(o)

    withProgress("Bulk extraction...", value = 0, {
      for (i in seq_along(fns)) {
        fname <- fns[[i]]
        incProgress(1/length(fns), detail = sprintf("%d/%d: %s", i, length(fns), fname))

        res <- o[[fname]]
        vec <- e[[fname]]; if (is.null(vec) || length(vec) != length(res$text_per_page)) vec <- res$text_per_page
        if (isTRUE(input$auto_spellcheck)) vec <- spellcheck_pages(vec)

        txt <- combine_pages(res$page_numbers, vec)

        out <- tryCatch(
          run_extraction(input$extract_model, fields, input$user_prompt, txt),
          error = function(err) { showNotification(paste(fname, ":", err$message), type="error"); NULL }
        )
        if (!is.null(out)) out_map[[fname]] <- out
      }
    })

    bulk_ext(out_map)
  })

  bulk_table <- reactive({
    ext <- bulk_ext()
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

  # ---- Downloads ---------------------------------------------------
  output$download_csv <- downloadHandler(
    filename = function() paste0(tools::file_path_sans_ext(input$file$name %||% "single"), "_fields.csv"),
    content  = function(file) write.csv(single_df() %||% data.frame(), file, row.names = FALSE)
  )

  output$download_json <- downloadHandler(
    filename = function() paste0(tools::file_path_sans_ext(input$file$name %||% "single"), "_fields.json"),
    content  = function(file) {
      df <- single_df()
      if (is.null(df)) return(writeLines("{}", file))
      out <- setNames(lapply(seq_len(nrow(df)), function(i) list(value=df$value[i], confidence=df$confidence[i])), df$name)
      write_json(out, file, pretty = TRUE, auto_unbox = TRUE)
    }
  )

  output$download_ocr_text <- downloadHandler(
    filename = function() paste0(tools::file_path_sans_ext(input$file$name %||% "single"), "_ocr.txt"),
    content  = function(file) {
      res <- single_res()
      if (is.null(res)) return(writeLines("No OCR result.", file))
      edt <- single_edits()
      txt <- if (!is.null(edt) && length(edt) == length(res$page_numbers)) combine_pages(res$page_numbers, edt) else res$combined_text
      writeLines(txt, file, useBytes = TRUE)
    }
  )

  output$download_bulk_csv <- downloadHandler(
    filename = function() paste0("bulk_results_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv"),
    content  = function(file) write.csv(bulk_table() %||% data.frame(), file, row.names = FALSE)
  )

  output$download_bulk_json <- downloadHandler(
    filename = function() paste0("bulk_results_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".json"),
    content  = function(file) {
      tab <- bulk_table()
      if (is.null(tab)) return(writeLines("{}", file))
      field_names <- tab$Field
      files <- setdiff(names(tab), "Field")
      out <- setNames(lapply(seq_along(field_names), function(i) {
        vals <- as.list(tab[i, files, drop = FALSE])
        lapply(vals, function(x) if (is.na(x)) "" else x)
      }), field_names)
      write_json(out, file, pretty = TRUE, auto_unbox = TRUE)
    }
  )
}

shinyApp(ui, server)
# ---- End of app.R --------------------------------------------------
