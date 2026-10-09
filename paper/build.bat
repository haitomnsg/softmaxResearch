@echo off
REM Build main.pdf. Requires pdflatex + bibtex on PATH (MiKTeX or TeX Live).
REM pdflatex -> bibtex -> pdflatex x2 resolves citations and cross-references.
REM Regenerate the figures first if the runs changed: python experiments\make_paper_figures.py
cd /d "%~dp0"
pdflatex -interaction=nonstopmode -halt-on-error main.tex || exit /b 1
bibtex main || exit /b 1
pdflatex -interaction=nonstopmode -halt-on-error main.tex || exit /b 1
pdflatex -interaction=nonstopmode -halt-on-error main.tex || exit /b 1
echo.
echo Built: main.pdf
