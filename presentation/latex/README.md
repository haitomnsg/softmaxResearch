# GAM-Softmax talk — LaTeX (Beamer) source

Two documents, one body of slides.

| file | what you get |
|---|---|
| `slides.tex` → `slides.pdf` | the deck you project — 31 pages: 21 content slides, 5 section dividers, 4 backup |
| `notes.tex` → `notes.pdf` | 57 pages: every slide followed by a page carrying its thumbnail and the paragraph to say |
| `content.tex` | every frame (edit here — both documents include it) |
| `preamble.tex` | theme, colours, macros |
| `figures/` | the four result plots, copied from `runs/final_20ng/figures/` |

Structure follows the required five sections: Introduction → Background →
Methodology → Experimentation → Results and Demonstration of Final Result.

## Build

### Option A — Overleaf (no install)

1. Zip this folder (`presentation/latex/`).
2. overleaf.com → New Project → Upload Project → drop the zip.
3. Menu → Compiler: **pdfLaTeX**, Main document: **slides.tex**. Recompile.
4. Switch the main document to `notes.tex` for the speaker script.

### Option B — locally, with MiKTeX or TeX Live

```bash
pdflatex slides.tex && pdflatex slides.tex
pdflatex notes.tex  && pdflatex notes.tex
```

Two passes: the slide counter in the footer (`n/24`) needs the first pass to
learn the total.

`build.bat` does both documents in one go on Windows.

Nothing exotic is used — `beamer`, `amsmath`, `amssymb`, `booktabs`,
`graphicx`, `xcolor`, `array`, `tikz`, `iftex`. Compiles with pdflatex,
xelatex, lualatex or tectonic.

## Editing

- **Numbers**: every figure in the deck comes from `FINAL_REPORT.md` §5 and
  `runs/final_20ng/summary.md`. If a number changes there, change it here.
- **Cutting for time**: 21 content slides is ~12–13 minutes at a comfortable
  pace. To hit a hard 10, drop *A worked example* (p. 6), *Every experiment in
  order* (p. 20), and *Limits and what a follow-up should do* (p. 27) — each is
  self-contained and nothing later depends on it.
- **Slide 16** (`let δ go negative`) is the one to slow down on. It is the only
  genuinely novel piece, and it is pure algebra, which is the part a maths
  audience will actually want to interrogate.
- **Speaker notes** live in `\note{...}` inside each frame in `content.tex`.
