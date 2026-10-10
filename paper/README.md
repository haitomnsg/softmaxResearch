# Paper draft — "When the Margin Goes Negative"

Phase E of [docs/10_noisy_label_plan.md](../docs/10_noisy_label_plan.md). The pre-registered Phase C′ rule
(P3 failed for both self-calibrating statistics) sent the project to a **reframing write-up** for TMLR or a
workshop instead of the CIFAR-N benchmark. This folder is that write-up.

| file | content |
|---|---|
| `main.tex` | preamble, title, section includes (plain `article` + natbib; swap in `tmlr.sty` / the workshop style at submission) |
| `sections/00_abstract.tex` … `06_discussion.tex` | abstract, intro, background + related work, theory, protocol, results, discussion + conclusion |
| `sections/appendix.tex` | proofs, the loss-mixture statistic, pre-registration record (Phases A–E), reproducibility |
| `refs.bib` | 37 entries, each checked against publisher / proceedings records on 2026-10-09 |
| `figures/` | generated — `python experiments/make_paper_figures.py` (PDF for LaTeX, PNG for preview) |

## Build

`build.bat` (pdflatex → bibtex → pdflatex ×2) → `main.pdf`, 20 pages; the PDF is committed. MiKTeX 25.12 is installed
per-user on the research desktop (`%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64`, packages auto-install on first
use); put that directory on `PATH` first if a shell predates the install. Elsewhere: Overleaf (zip this folder, compiler
pdfLaTeX, main document `main.tex`) or any TeX Live. Regenerate figures before building if the runs changed:
`python experiments/make_paper_figures.py`.

## Where every number comes from

| paper element | source |
|---|---|
| Tables 1–2, Figs 2–4 (`figC1`, `figC2`, `fig4`) | `runs/phase_c/*.json` (5 seeds) → `runs/phase_c/summary.md` |
| Table 3 (verdicts) | docs/10 §6 status table, which records each verdict when it was made |
| Appendix Table (Phases A–B′) | `runs/final_20ng/summary.md` |
| Fig 1 (geometry), Corollary 2 band values | analytic; checked against the loss in `tests/test_mechanism.py` |
| §5.4 "Where the gradient goes", Fig 5 | `runs/mechanism/` (`experiments/mechanism.py`, 42 runs, 3 seeds) → `runs/mechanism/summary.md` |

`runs/` is gitignored, so the JSONs live only on the research machine.

## Open items

- [x] §5.4 mechanism paragraph + `fig5_mechanism` (2026-10-10).
- [x] Appendix Table "Phases": the Phase E outcome cell.
- [x] First compile (2026-10-09): clean, no undefined references or citations.
- [ ] Author block (anonymous for double-blind review) and the venue's style file.
- [ ] Decision for the authors: the escape-region prediction (β = 1.15 closes the band to [0.032, 1.0)) is stated
      but not run, because β was frozen before Phase C′. Running it would be a clearly labelled post-hoc experiment.
