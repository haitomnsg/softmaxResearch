@echo off
REM Build both PDFs. Requires pdflatex on PATH (MiKTeX or TeX Live).
REM Two passes each: the footer slide counter needs the total from pass 1.
cd /d "%~dp0"
pdflatex -interaction=nonstopmode -halt-on-error slides.tex || exit /b 1
pdflatex -interaction=nonstopmode -halt-on-error slides.tex || exit /b 1
pdflatex -interaction=nonstopmode -halt-on-error notes.tex  || exit /b 1
pdflatex -interaction=nonstopmode -halt-on-error notes.tex  || exit /b 1
echo.
echo Built: slides.pdf and notes.pdf
