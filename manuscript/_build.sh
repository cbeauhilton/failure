#!/bin/sh

set -ev

Rscript -e "bookdown::render_book('index.rmd', 'bookdown::html_document2')"
mv _main.html _book/one_page.html
Rscript -e "bookdown::render_book('index.rmd', 'bookdown::gitbook')"
Rscript -e "bookdown::render_book('index.rmd', 'bookdown::pdf_book')"
Rscript -e "bookdown::render_book('index.rmd', 'bookdown::epub_book')"
