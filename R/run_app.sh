#!/usr/bin/env bash
set -e

cd /home/doanm/ocr-automate/R 

# If you rely on bash environment variables
source ~/.bashrc


/usr/bin/Rscript -e "shiny::runApp('/home/$USER/ocr-automate/R', host = '0.0.0.0', port = 8501, launch.browser = FALSE)"
