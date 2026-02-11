# extract_module.R -----------------------------------------------------
library(shiny)
library(jsonlite)
library(ollamar)

# ---- JSON helper for extraction -------------------------------------
extract_json_substring <- function(x) {
  start <- regexpr("\\{", x)
  end   <- regexpr(".*\\}", x)
  if (start[1] == -1 || end[1] == -1) return(x)
  substr(x, start[1], start[1] + attr(end, "match.length") - 1)
}

# ---- Spellcheck ------------------------------------------------------
spellcheck_pages <- function(pages, notify_fn = NULL) {
  if (!requireNamespace("hunspell", quietly = TRUE)) {
    if (!is.null(notify_fn)) notify_fn("Package 'hunspell' not installed; skipping spellcheck.", "warning")
    return(pages)
  }

  out <- pages
  for (i in seq_along(pages)) {
    txt <- pages[[i]]
    if (is.null(txt) || !nzchar(txt)) next

    bad_list <- hunspell::hunspell(txt)[[1]]
    if (length(bad_list) == 0) next

    for (w in bad_list) {
      sugg <- hunspell::hunspell_suggest(w)[[1]]
      if (length(sugg) == 1) {
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

# ---- Prompt builder + model call ------------------------------------
run_extraction_on_text <- function(
  ocr_text_full,
  field_list,
  user_prompt,
  extract_model,
  max_chars = 32000L
) {
  fields <- strsplit(field_list, "\\r?\\n")[[1]]
  fields <- trimws(fields)
  fields <- fields[nzchar(fields)]
  if (length(fields) == 0) stop("Please provide at least one field name.")

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

  ocr_text <- ocr_text_full
  if (!is.null(ocr_text) && nchar(ocr_text) > max_chars) {
    cat("[DEBUG] OCR text truncated from", nchar(ocr_text), "to", max_chars, "characters\n")
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

  model <- trimws(extract_model)
  resp_text <- NULL

  if (identical(model, "gpt-oss")) {
    full_prompt <- paste(system_msg, "\n\n---\n\n", user_msg, sep = "")
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

  if (is.null(parsed$fields)) stop("Response JSON does not contain 'fields' key.")

  df <- as.data.frame(parsed$fields, stringsAsFactors = FALSE)
  names(df) <- tolower(names(df))
  if (!"name" %in% names(df)) df$name <- NA_character_
  if (!"value" %in% names(df)) df$value <- NA_character_
  if (!"confidence" %in% names(df)) df$confidence <- NA_real_

  list(df = df, raw = resp_text)
}

# ---- Server wiring (single + bulk) ----------------------------------
# ocr_state: list(
#   ocr_result = reactiveVal(...),
#   edited_page_texts = reactiveVal(...),
#   bulk_ocr_results = reactiveVal(...),
#   bulk_edited_page_texts = reactiveVal(...)
# )
extract_server <- function(input, output, session, ocr_state) {

  extract_df   <- reactiveVal(NULL)
  extract_json <- reactiveVal(NULL)

  bulk_extract_results <- reactiveVal(list())

  observeEvent(input$run_extract, {
    # Optional uploaded OCR text (bypasses need for run_ocr)
    uploaded_text <- NULL
    if (!is.null(input$extract_text_file) && nrow(input$extract_text_file) > 0) {
      uploaded_text <- paste(
        readLines(input$extract_text_file$datapath, warn = FALSE, encoding = "UTF-8"),
        collapse = "\n"
      )
    }

    res <- ocr_state$ocr_result()

    if (is.null(uploaded_text) || !nzchar(uploaded_text)) {
      req(res)
    }

    withProgress(
      message = "Running field extraction with gpt-oss...",
      value = 0,
      {
        incProgress(0.2, detail = "Calling Ollama…")

        out <- tryCatch(
          {
            if (!is.null(uploaded_text) && nzchar(uploaded_text)) {
              ocr_text <- uploaded_text
            } else {
              edt <- ocr_state$edited_page_texts()
              page_nums <- if (!is.null(res$page_numbers)) {
                res$page_numbers
              } else if (!is.null(res$text_per_page)) {
                seq_along(res$text_per_page)
              } else {
                NULL
              }

              if (!is.null(edt) && isTRUE(input$auto_spellcheck)) {
                edt_checked <- spellcheck_pages(
                  edt,
                  notify_fn = function(msg, type) showNotification(msg, type = type)
                )
              } else {
                edt_checked <- edt
              }

              if (!is.null(edt_checked) && !is.null(page_nums) &&
                  length(edt_checked) == length(page_nums)) {
                ocr_text <- paste(
                  sprintf("---- PAGE %d ----\n%s", page_nums, edt_checked),
                  collapse = "\n\n"
                )
              } else {
                ocr_text <- res$combined_text
              }
            }

            run_extraction_on_text(
              ocr_text_full  = ocr_text,
              field_list     = input$field_list,
              user_prompt    = input$user_prompt,
              extract_model  = input$extract_model,
              max_chars      = 32000L
            )
          },
          error = function(e) {
            showNotification(paste("Error during extraction:", e$message), type = "error")
            NULL
          }
        )

        if (!is.null(out)) {
          extract_df(out$df)
          extract_json(out$raw)
        }
      }
    )

    updateTabsetPanel(session, "tabs", selected = "Extracted Fields")
  })

  observeEvent(input$run_bulk_extract, {
    ocr_list <- ocr_state$bulk_ocr_results()
    req(length(ocr_list) > 0)

    withProgress(
      message = "Running bulk extraction...",
      value = 0,
      {
        file_names <- names(ocr_list)
        n_files <- length(file_names)
        ext_res_list <- list()
        edt_list <- ocr_state$bulk_edited_page_texts()

        for (i in seq_len(n_files)) {
          fname <- file_names[i]
          incProgress(amount = 1 / n_files, detail = sprintf("File %d of %d: %s", i, n_files, fname))

          ocr_res <- ocr_list[[fname]]
          if (is.null(ocr_res$text_per_page)) next

          page_vec <- if (!is.null(edt_list) &&
                          !is.null(edt_list[[fname]]) &&
                          length(edt_list[[fname]]) == length(ocr_res$text_per_page)) {
            edt_list[[fname]]
          } else {
            ocr_res$text_per_page
          }

          page_nums <- if (!is.null(ocr_res$page_numbers)) ocr_res$page_numbers else seq_along(page_vec)

          combined_text <- paste(
            sprintf("---- PAGE %d ----\n%s", page_nums, page_vec),
            collapse = "\n\n"
          )

          ext_res <- tryCatch(
            {
              run_extraction_on_text(
                ocr_text_full  = combined_text,
                field_list     = input$field_list,
                user_prompt    = input$user_prompt,
                extract_model  = input$extract_model,
                max_chars      = 32000L
              )
            },
            error = function(e) {
              showNotification(paste("Bulk extraction error for", fname, ":", e$message), type = "error")
              NULL
            }
          )

          if (!is.null(ext_res)) ext_res_list[[fname]] <- ext_res
        }

        bulk_extract_results(ext_res_list)
      }
    )
  })

  # expose reactives back to main app
  list(
    extract_df = extract_df,
    extract_json = extract_json,
    bulk_extract_results = bulk_extract_results
  )
}
