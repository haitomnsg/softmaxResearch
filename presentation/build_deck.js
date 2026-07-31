// Builds the 10-minute GAM-Softmax presentation.
//   node build_deck.js
// Numbers come from runs/final_20ng/summary.md (2 seeds) and FINAL_REPORT.md.

const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
pres.author = "Ashish Gupta";
pres.title = "GAM-Softmax";

// ---------------------------------------------------------------- palette
const NAVY = "152238"; // dominant dark
const NAVY_SOFT = "24354F"; // dark card fill
const WHITE = "FFFFFF";
const CHERRY = "C1121F"; // the problem: memorization, failed hypothesis
const TEAL = "12897B"; // what worked: our sample margin
const AMBER = "C97B1E"; // partial / the variant that didn't work
const BODY = "44506A"; // body text on light
const MUTED = "8A93A6"; // captions
const CARD = "EEF2F7"; // light card fill
const CARD_EDGE = "D8E0EA";

const HEAD = "Cambria"; // safe-list serif
const TEXT = "Calibri"; // safe-list sans

// fresh object every call — pptxgenjs mutates options in place
const shadow = (o = {}) => ({
  type: "outer",
  color: "9AA6B8",
  blur: 10,
  offset: 2,
  angle: 90,
  opacity: 0.28,
  ...o,
});

// ------------------------------------------------------------- primitives
function lightSlide(title, kicker) {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  if (kicker) {
    s.addText(kicker.toUpperCase(), {
      x: 0.6, y: 0.34, w: 12.1, h: 0.28,
      fontFace: TEXT, fontSize: 12, bold: true, color: TEAL, charSpacing: 2, margin: 0,
    });
  }
  s.addText(title, {
    x: 0.6, y: kicker ? 0.66 : 0.5, w: 12.1, h: 0.75,
    fontFace: HEAD, fontSize: 32, bold: true, color: NAVY, margin: 0, valign: "top",
  });
  return s;
}

function darkSlide(title, kicker) {
  const s = pres.addSlide();
  s.background = { color: NAVY };
  if (kicker) {
    s.addText(kicker.toUpperCase(), {
      x: 0.6, y: 0.34, w: 12.1, h: 0.28,
      fontFace: TEXT, fontSize: 12, bold: true, color: "6FD8C9", charSpacing: 2, margin: 0,
    });
  }
  s.addText(title, {
    x: 0.6, y: kicker ? 0.66 : 0.5, w: 12.1, h: 0.75,
    fontFace: HEAD, fontSize: 32, bold: true, color: WHITE, margin: 0, valign: "top",
  });
  return s;
}

// rounded card with optional icon-in-circle motif (the deck's repeating element)
function card(s, { x, y, w, h, fill = CARD, line = CARD_EDGE }) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.12,
    fill: { color: fill },
    line: { color: line, width: 1 },
    shadow: shadow(),
  });
}

function circleBadge(s, { x, y, d = 0.52, fill, label, labelColor = WHITE, size = 17 }) {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
  s.addText(label, {
    x, y, w: d, h: d, align: "center", valign: "middle", margin: 0,
    fontFace: TEXT, fontSize: size, bold: true, color: labelColor,
  });
}

function statBlock(s, { x, y, w, value, label, color = TEAL, valueSize = 40 }) {
  s.addText(value, {
    x, y, w, h: 0.7, margin: 0, align: "left",
    fontFace: HEAD, fontSize: valueSize, bold: true, color,
  });
  s.addText(label, {
    x, y: y + 0.68, w, h: 0.62, margin: 0, align: "left",
    fontFace: TEXT, fontSize: 12.5, color: BODY,
  });
}

// horizontal probability bar used in the "what a classifier does" visuals
function probBar(s, { x, y, w, label, pct, color, maxPct = 100, textColor = BODY }) {
  s.addText(label, {
    x, y, w: 1.45, h: 0.3, margin: 0, valign: "middle",
    fontFace: TEXT, fontSize: 12, color: textColor,
  });
  const trackX = x + 1.5;
  const trackW = w - 1.5 - 0.75;
  s.addShape(pres.ShapeType.roundRect, {
    x: trackX, y: y + 0.05, w: trackW, h: 0.22, rectRadius: 0.11,
    fill: { color: "E3E8EF" }, line: { color: "E3E8EF" },
  });
  const fillW = Math.max(0.06, (trackW * pct) / maxPct);
  s.addShape(pres.ShapeType.roundRect, {
    x: trackX, y: y + 0.05, w: fillW, h: 0.22, rectRadius: 0.11,
    fill: { color }, line: { color },
  });
  s.addText(`${pct}%`, {
    x: trackX + trackW + 0.06, y, w: 0.7, h: 0.3, margin: 0, valign: "middle",
    fontFace: TEXT, fontSize: 11.5, bold: true, color: textColor,
  });
}

// =====================================================================  1
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  s.addText("δ", {
    x: 8.5, y: 3.95, w: 4.4, h: 1.9, margin: 0, align: "right",
    fontFace: HEAD, fontSize: 118, bold: true, color: "1E3050",
  });
  s.addText("GAM-Softmax", {
    x: 0.85, y: 2.0, w: 11.6, h: 1.15, margin: 0,
    fontFace: HEAD, fontSize: 56, bold: true, color: WHITE,
  });
  s.addText("Can a smarter margin beat plain cross-entropy?", {
    x: 0.85, y: 3.15, w: 11.6, h: 0.6, margin: 0,
    fontFace: HEAD, fontSize: 25, color: "6FD8C9",
  });
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.85, y: 4.12, w: 5.2, h: 0.62, rectRadius: 0.14,
    fill: { color: NAVY_SOFT }, line: { color: "35496B", width: 1 },
  });
  s.addText("p_t  −  p_j   ≥   δ", {
    x: 0.85, y: 4.12, w: 5.2, h: 0.62, align: "center", valign: "middle", margin: 0,
    fontFace: TEXT, fontSize: 19, bold: true, color: WHITE,
  });
  s.addText("Generalising the margin of AS-Softmax along class-pair, sample and time", {
    x: 0.85, y: 5.05, w: 11.6, h: 0.35, margin: 0,
    fontFace: TEXT, fontSize: 14.5, color: "A9B6CC",
  });
  s.addText("Ashish Gupta   ·   End-of-semester research report   ·   July 2026", {
    x: 0.85, y: 6.35, w: 11.6, h: 0.35, margin: 0,
    fontFace: TEXT, fontSize: 13, color: "7C8AA5",
  });
  s.addNotes(
    "[30s   ·   clock 0:30]\n\n" +
    "Hi everyone. This is a research project on classification loss functions.\n\n" +
    "The one-line version: there's a published method that replaces the usual training " +
    "objective with a 'margin' rule. We tried to make that margin smarter. I'll tell you " +
    "what we tried, what failed, and the one thing that actually worked.\n\n" +
    "I'm going to start from the absolute basics, so nobody gets lost."
  );
}

// =====================================================================  2
{
  const s = lightSlide("A classifier is a student taking a multiple-choice test", "The basics");

  s.addText(
    "For every input, the model gives each possible answer a score. " +
    "Softmax turns those raw scores into probabilities that add up to 100%.",
    { x: 0.6, y: 1.6, w: 6.1, h: 0.9, margin: 0, fontFace: TEXT, fontSize: 15.5, color: BODY, lineSpacing: 22 }
  );

  card(s, { x: 0.6, y: 2.9, w: 6.1, h: 2.0 });
  s.addText("The question we actually care about at test time:", {
    x: 0.9, y: 3.2, w: 5.5, h: 0.3, margin: 0, fontFace: TEXT, fontSize: 13, color: MUTED,
  });
  s.addText("Is the right answer simply AHEAD of the others?", {
    x: 0.9, y: 3.65, w: 5.5, h: 0.4, margin: 0, fontFace: HEAD, fontSize: 18, bold: true, color: NAVY,
  });
  s.addText("Not: is it at 100%?", {
    x: 0.9, y: 4.2, w: 5.5, h: 0.35, margin: 0, fontFace: TEXT, fontSize: 15, italic: true, color: CHERRY,
  });

  s.addText("Remember that gap — the whole project lives there.", {
    x: 0.6, y: 5.25, w: 6.1, h: 0.4, margin: 0, fontFace: TEXT, fontSize: 13.5, color: MUTED, italic: true,
  });

  // right: worked example
  card(s, { x: 7.1, y: 1.6, w: 5.6, h: 4.45, fill: WHITE, line: CARD_EDGE });
  s.addText('Review: "a dull, lifeless film"', {
    x: 7.45, y: 1.95, w: 4.9, h: 0.35, margin: 0, fontFace: TEXT, fontSize: 13.5, italic: true, color: NAVY,
  });
  s.addText("The model's answer, as probabilities", {
    x: 7.45, y: 2.35, w: 4.9, h: 0.3, margin: 0, fontFace: TEXT, fontSize: 11.5, color: MUTED,
  });
  const rows = [
    ["very negative", 55, TEAL],
    ["negative", 28, "9AA6B8"],
    ["neutral", 11, "9AA6B8"],
    ["positive", 4, "9AA6B8"],
    ["very positive", 2, "9AA6B8"],
  ];
  rows.forEach(([lab, pct, col], i) =>
    probBar(s, { x: 7.45, y: 2.85 + i * 0.56, w: 4.9, label: lab, pct, color: col, maxPct: 60 })
  );
  s.addText("It is already right — the top answer is ahead.", {
    x: 7.45, y: 5.6, w: 4.9, h: 0.3, margin: 0, fontFace: TEXT, fontSize: 12, bold: true, color: TEAL,
  });

  s.addNotes(
    "[35s   ·   clock 1:05]\n\n" +
    "Simplest possible framing. A classifier is a student answering multiple-choice questions.\n\n" +
    "For each question it scores every option, and softmax squashes those scores into " +
    "percentages that sum to 100.\n\n" +
    "Here the model reads a bad film review and puts 55% on 'very negative'. It's right — " +
    "because 'very negative' is AHEAD of everything else.\n\n" +
    "Key point to plant now: being right only requires being ahead. It does NOT require 100%. " +
    "That gap is where this entire project lives."
  );
}

// =====================================================================  3
{
  const s = lightSlide("But normal training demands 100% — forever", "The problem");

  s.addText(
    "Cross-entropy, the standard training objective, keeps pushing the correct answer " +
    "toward 100% and every wrong answer toward 0% — even on examples the model already gets right.",
    { x: 0.6, y: 1.6, w: 12.1, h: 0.75, margin: 0, fontFace: TEXT, fontSize: 15.5, color: BODY, lineSpacing: 22 }
  );

  card(s, { x: 0.6, y: 2.6, w: 5.9, h: 2.95, fill: WHITE });
  s.addText("Already correct at epoch 1", {
    x: 0.95, y: 2.85, w: 5.2, h: 0.3, margin: 0, fontFace: TEXT, fontSize: 12.5, bold: true, color: MUTED,
  });
  probBar(s, { x: 0.95, y: 3.35, w: 5.2, label: "very negative", pct: 55, color: TEAL, maxPct: 100 });
  probBar(s, { x: 0.95, y: 3.95, w: 5.2, label: "negative", pct: 28, color: "9AA6B8", maxPct: 100 });
  s.addText("Test would already score this correct.", {
    x: 0.95, y: 4.75, w: 5.2, h: 0.55, margin: 0, fontFace: TEXT, fontSize: 12.5, color: BODY,
  });

  card(s, { x: 6.8, y: 2.6, w: 5.9, h: 2.95, fill: "FBEDEE", line: "F0CFD2" });
  s.addText("Cross-entropy keeps going anyway", {
    x: 7.15, y: 2.85, w: 5.2, h: 0.3, margin: 0, fontFace: TEXT, fontSize: 12.5, bold: true, color: CHERRY,
  });
  probBar(s, { x: 7.15, y: 3.35, w: 5.2, label: "very negative", pct: 99, color: CHERRY, maxPct: 100 });
  probBar(s, { x: 7.15, y: 3.95, w: 5.2, label: "negative", pct: 1, color: "E3A3A8", maxPct: 100 });
  s.addText("All this extra work buys zero extra correctness.", {
    x: 7.15, y: 4.75, w: 5.2, h: 0.55, margin: 0, fontFace: TEXT, fontSize: 12.5, color: CHERRY,
  });

  const costs = [
    ["Wasted effort", "training on answers it already knows"],
    ["Overfitting", "it starts memorising instead of learning"],
    ["Loss stops tracking accuracy", "the number you watch stops meaning anything"],
  ];
  costs.forEach(([h, d], i) => {
    const x = 0.6 + i * 4.06;
    circleBadge(s, { x, y: 5.95, fill: CHERRY, label: String(i + 1) });
    s.addText(h, {
      x: x + 0.66, y: 5.95, w: 3.25, h: 0.3, margin: 0,
      fontFace: TEXT, fontSize: 13.5, bold: true, color: NAVY,
    });
    s.addText(d, {
      x: x + 0.66, y: 6.26, w: 3.25, h: 0.55, margin: 0,
      fontFace: TEXT, fontSize: 11.5, color: MUTED,
    });
  });

  s.addNotes(
    "[40s   ·   clock 1:45]\n\n" +
    "Here's the problem. Standard training — cross-entropy — is never satisfied.\n\n" +
    "Left: the model is already right, 55% on the correct answer. Test passed.\n\n" +
    "Right: cross-entropy keeps pushing anyway, to 99%. That extra work buys you nothing " +
    "at test time, because the test only asked 'is it ahead?'\n\n" +
    "Three costs: wasted effort; overfitting, because to reach 99% on hard examples it " +
    "starts memorising them; and the loss stops correlating with accuracy."
  );
}

// =====================================================================  4
{
  const s = lightSlide("The paper we build on: AS-Softmax (Lv et al., 2023)", "Prior work");

  card(s, { x: 0.6, y: 1.55, w: 12.1, h: 1.5, fill: "E9F5F3", line: "BFE0DA" });
  s.addText("Their rule:", {
    x: 0.95, y: 1.86, w: 1.6, h: 0.35, margin: 0, fontFace: TEXT, fontSize: 14, bold: true, color: TEAL, valign: "middle",
  });
  s.addText("if   p_t  −  p_j   ≥   δ     →   stop training against wrong answer j", {
    x: 2.4, y: 1.8, w: 9.9, h: 0.45, margin: 0, fontFace: TEXT, fontSize: 19, bold: true, color: NAVY, valign: "middle",
  });
  s.addText(
    "Once the right answer beats a wrong one by a fixed gap δ, that wrong answer is dropped from the loss. " +
    "Like a student who only needs to pass comfortably — not score 100 on every question.",
    { x: 0.95, y: 2.34, w: 11.4, h: 0.6, margin: 0, fontFace: TEXT, fontSize: 13.5, color: BODY, lineSpacing: 18 }
  );

  s.addText("What they reported", {
    x: 0.6, y: 3.4, w: 6.0, h: 0.32, margin: 0, fontFace: TEXT, fontSize: 13, bold: true, color: MUTED,
  });
  const wins = [
    ["Better accuracy & F1", "across most tasks"],
    ["Better calibration, less overfitting", ""],
    ["≈1.2× faster training", "via their AS-Speed trick"],
  ];
  wins.forEach(([h, d], i) => {
    const y = 3.85 + i * 0.72;
    circleBadge(s, { x: 0.6, y, d: 0.36, fill: TEAL, label: "✓", size: 13 });
    s.addText(d ? `${h} — ${d}` : h, {
      x: 1.08, y, w: 5.5, h: 0.36, margin: 0, valign: "middle",
      fontFace: TEXT, fontSize: 13.5, color: NAVY,
    });
  });

  card(s, { x: 6.9, y: 3.7, w: 5.8, h: 1.8, fill: WHITE });
  s.addText("Their headline number (SST-5)", {
    x: 7.2, y: 3.9, w: 5.2, h: 0.28, margin: 0, fontFace: TEXT, fontSize: 11.5, color: MUTED,
  });
  s.addText("0.038   →   −0.952", {
    x: 7.2, y: 4.2, w: 5.2, h: 0.5, margin: 0, fontFace: HEAD, fontSize: 26, bold: true, color: TEAL,
  });
  s.addText("correlation between loss and validation accuracy: the loss finally tracks what we care about", {
    x: 7.2, y: 4.73, w: 5.2, h: 0.55, margin: 0, fontFace: TEXT, fontSize: 11, color: BODY,
  });

  s.addText("Tested on:   SST-5 · CLINC150 · CoNLL2003 · SIGHAN2015 · Eurlex · WOS-46985   (text)   ·   WikiArt   (images)   ·   Chest-falsetto   (audio)", {
    x: 0.6, y: 5.85, w: 12.1, h: 0.34, margin: 0, fontFace: TEXT, fontSize: 12, color: MUTED,
  });
  s.addText("The opening they leave: δ is a single global number, hand-tuned per dataset.", {
    x: 0.6, y: 6.32, w: 12.1, h: 0.36, margin: 0, fontFace: TEXT, fontSize: 14.5, bold: true, italic: true, color: CHERRY,
  });

  s.addNotes(
    "[45s   ·   clock 2:30]\n\n" +
    "This is the paper we build on — AS-Softmax, 2023.\n\n" +
    "Their fix: don't demand 100%, demand a gap. Once the right answer beats a wrong one " +
    "by delta, drop that wrong answer and stop pushing.\n\n" +
    "They reported better accuracy, better calibration, less overfitting, about 1.2x faster " +
    "training — and this striking number: the correlation between loss and validation " +
    "accuracy went from basically zero to minus 0.95. The loss finally tracks quality.\n\n" +
    "Tested broadly — six text datasets, images, audio.\n\n" +
    "And here is the door they leave open, our starting point: delta is ONE number, for " +
    "every class pair, every example, every moment of training."
  );
}

// =====================================================================  5
{
  const s = lightSlide("Our idea: one number for everything is too crude", "Our approach");

  s.addShape(pres.ShapeType.roundRect, {
    x: 3.55, y: 1.5, w: 6.2, h: 0.72, rectRadius: 0.14,
    fill: { color: NAVY }, line: { color: NAVY },
  });
  s.addText("δ      →      δ t,j (x, τ)", {
    x: 3.55, y: 1.5, w: 6.2, h: 0.72, align: "center", valign: "middle", margin: 0,
    fontFace: TEXT, fontSize: 21, bold: true, color: WHITE,
  });
  s.addText("one fixed gap  →  a gap that adapts", {
    x: 3.55, y: 2.3, w: 6.2, h: 0.3, align: "center", margin: 0,
    fontFace: TEXT, fontSize: 12.5, italic: true, color: MUTED,
  });

  const axes = [
    ["1", "Class-pair", "Some answers are genuinely easy to confuse.", '"cat vs dog" deserves a bigger gap than "cat vs aeroplane"', TEAL],
    ["2", "Sample", "Some individual examples are ambiguous or plain wrong.", "A blurry photo should not be treated like a clear one", AMBER],
    ["3", "Time", "Early training is basically guessing.", "Start loose, tighten as the model becomes trustworthy", "3E6FA8"],
  ];
  axes.forEach(([n, title, lead, ex, col], i) => {
    const x = 0.6 + i * 4.06;
    card(s, { x, y: 2.9, w: 3.72, h: 3.15, fill: WHITE });
    circleBadge(s, { x: x + 0.32, y: 3.18, d: 0.52, fill: col, label: n });
    s.addText(title, {
      x: x + 0.98, y: 3.2, w: 2.5, h: 0.45, margin: 0, valign: "middle",
      fontFace: HEAD, fontSize: 19, bold: true, color: NAVY,
    });
    s.addText(lead, {
      x: x + 0.32, y: 3.95, w: 3.1, h: 0.8, margin: 0,
      fontFace: TEXT, fontSize: 13, color: BODY, lineSpacing: 18,
    });
    s.addText(ex, {
      x: x + 0.32, y: 4.9, w: 3.1, h: 0.95, margin: 0,
      fontFace: TEXT, fontSize: 11.5, italic: true, color: MUTED, lineSpacing: 16,
    });
  });

  s.addText("We test one axis at a time, so any gain can be traced to a specific cause.", {
    x: 0.6, y: 6.35, w: 12.1, h: 0.34, margin: 0, align: "center",
    fontFace: TEXT, fontSize: 13.5, italic: true, color: BODY,
  });

  s.addNotes(
    "[45s   ·   clock 3:15]\n\n" +
    "So here's our idea. If one gap is good, a gap that ADAPTS should be better.\n\n" +
    "Three directions it could adapt in.\n\n" +
    "One, class-pair: cat versus dog is a harder distinction than cat versus aeroplane, so " +
    "they shouldn't share a gap.\n\n" +
    "Two, sample: individual examples differ. Some are ambiguous, some are just wrong.\n\n" +
    "Three, time: at the start the model is guessing, so a strict gap is dangerous. Start " +
    "loose and tighten.\n\n" +
    "Important methodological point: we test these ONE AT A TIME, so if something improves " +
    "we know exactly what caused it."
  );
}

// =====================================================================  6
{
  const s = lightSlide("What we built", "Engineering");

  const stats = [
    ["9", "loss functions implemented\nand unit-tested"],
    ["70", "tests passing\nin under 3 seconds"],
    ["3", "margin variants\n(class-pair, sample, combined)"],
    ["6 GB", "one laptop GPU —\nthe entire compute budget"],
  ];
  stats.forEach(([v, l], i) => {
    const x = 0.6 + i * 3.09;
    card(s, { x, y: 1.6, w: 2.83, h: 1.95, fill: WHITE });
    statBlock(s, { x: x + 0.3, y: 1.95, w: 2.3, value: v, label: l, color: i === 3 ? AMBER : TEAL, valueSize: 34 });
  });

  card(s, { x: 0.6, y: 3.85, w: 6.1, h: 2.85, fill: CARD });
  s.addText("Built so the result is checkable", {
    x: 0.95, y: 4.1, w: 5.4, h: 0.35, margin: 0, fontFace: HEAD, fontSize: 18, bold: true, color: NAVY,
  });
  s.addText(
    [
      { text: "One config file = one experiment = one row in the results table", options: { bullet: true, breakLine: true } },
      { text: "A test proves our loss reduces exactly to AS-Softmax when the gap is held constant", options: { bullet: true, breakLine: true } },
      { text: "Adding a new idea = one new file + one YAML, not a rewrite", options: { bullet: true } },
    ],
    { x: 0.95, y: 4.62, w: 5.4, h: 1.85, margin: 0, fontFace: TEXT, fontSize: 13, color: BODY, paraSpaceAfter: 10, lineSpacing: 18 }
  );

  card(s, { x: 6.9, y: 3.85, w: 5.8, h: 2.85, fill: WHITE });
  s.addText("How we tested it", {
    x: 7.25, y: 4.1, w: 5.1, h: 0.35, margin: 0, fontFace: HEAD, fontSize: 18, bold: true, color: NAVY,
  });
  const setup = [
    ["Model", "BERT-base"],
    ["Datasets", "SST-5 (5 classes) · 20 Newsgroups (20 classes)"],
    ["Compared against", "plain cross-entropy and AS-Softmax"],
    ["Success bar, set in advance", "beat AS-Softmax by ≥ 0.3% accuracy"],
  ];
  setup.forEach(([k, v], i) => {
    const y = 4.68 + i * 0.5;
    s.addText(k, {
      x: 7.25, y, w: 2.3, h: 0.3, margin: 0, fontFace: TEXT, fontSize: 11.5, bold: true, color: MUTED, valign: "middle",
    });
    s.addText(v, {
      x: 9.35, y, w: 3.0, h: 0.3, margin: 0, fontFace: TEXT, fontSize: 11.5, color: NAVY, valign: "middle",
    });
  });

  s.addNotes(
    "[40s   ·   clock 3:55]\n\n" +
    "Briefly, what we built — because the result only matters if the code is trustworthy.\n\n" +
    "Nine loss functions, seventy passing tests, three margin variants, all on one 6 GB " +
    "laptop GPU.\n\n" +
    "Two things worth noting. There is a test proving our loss becomes EXACTLY AS-Softmax " +
    "when the gap is held constant — so we always compare a strict generalisation, not a " +
    "different loss. And we set the success bar in advance: beat AS-Softmax by 0.3%. " +
    "Setting that before you see results is what stops you fooling yourself."
  );
}

// =====================================================================  7
{
  const s = darkSlide("Then we ran it. It did not work.", "Results, part 1");

  s.addText(
    "Four testbeds. On accuracy, nothing separated — every difference was smaller than the " +
    "random variation between two runs of the same method.",
    { x: 0.6, y: 1.55, w: 12.1, h: 0.6, margin: 0, fontFace: TEXT, fontSize: 15, color: "C3CEDF", lineSpacing: 21 }
  );

  const rows = [
    ["SST-5, clean labels", "class-pair gap vs AS-Softmax", "−0.33%", "no gain"],
    ["SST-5, tuning screen", "let the gap learn harder", "−2.4%", "actively worse"],
    ["20 Newsgroups, clean", "class-pair gap", "+0.6%", "but plain CE still won"],
    ["SST-5, noisy labels", "class-pair gap, 3 noise levels", "±2%", "all inside seed noise"],
  ];
  rows.forEach(([test, what, num, verdict], i) => {
    const y = 2.35 + i * 0.72;
    s.addShape(pres.ShapeType.roundRect, {
      x: 0.6, y, w: 12.1, h: 0.6, rectRadius: 0.1,
      fill: { color: NAVY_SOFT }, line: { color: "35496B", width: 1 },
    });
    s.addText(test, {
      x: 0.9, y, w: 3.2, h: 0.6, margin: 0, valign: "middle",
      fontFace: TEXT, fontSize: 13.5, bold: true, color: WHITE,
    });
    s.addText(what, {
      x: 4.15, y, w: 4.2, h: 0.6, margin: 0, valign: "middle",
      fontFace: TEXT, fontSize: 12.5, color: "A9B6CC",
    });
    s.addText(num, {
      x: 8.45, y, w: 1.5, h: 0.6, margin: 0, valign: "middle", align: "right",
      fontFace: HEAD, fontSize: 17, bold: true, color: CHERRY === "C1121F" ? "FF8A8F" : CHERRY,
    });
    s.addText(verdict, {
      x: 10.2, y, w: 2.35, h: 0.6, margin: 0, valign: "middle",
      fontFace: TEXT, fontSize: 12, italic: true, color: "8A97AD",
    });
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: 5.42, w: 12.1, h: 1.25, rectRadius: 0.12,
    fill: { color: "3A1216" }, line: { color: "6B2229", width: 1 },
  });
  s.addText("Why? The problem was underneath us, not in our idea.", {
    x: 0.95, y: 5.58, w: 11.4, h: 0.34, margin: 0,
    fontFace: HEAD, fontSize: 17, bold: true, color: "FFB3B7",
  });
  s.addText(
    "AS-Softmax — the published method we were extending — did not reliably beat plain cross-entropy in our setup either. " +
    "You cannot improve on an advantage that isn't there.",
    { x: 0.95, y: 5.95, w: 11.4, h: 0.6, margin: 0, fontFace: TEXT, fontSize: 13.5, color: "F0D2D4", lineSpacing: 18 }
  );

  s.addNotes(
    "[50s   ·   clock 4:45]\n\n" +
    "Then we ran it, and it did not work. I want to be straight about that.\n\n" +
    "Four testbeds. Our class-pair gap never beat AS-Softmax by more than the noise between " +
    "two runs of the same method. We checked whether we had simply under-tuned it — we let " +
    "the gap learn harder and it got monotonically WORSE. So not a tuning problem.\n\n" +
    "The diagnosis is at the bottom, and it is the real finding. The problem was not our " +
    "idea, it was underneath us: AS-Softmax itself did not reliably beat plain cross-entropy " +
    "in our hands. We were generalising an advantage that was not there.\n\n" +
    "That is where a lot of projects quietly stop. We changed the question instead."
  );
}

// =====================================================================  8
{
  const s = lightSlide("So we changed the question", "The pivot");

  card(s, { x: 0.6, y: 1.55, w: 5.85, h: 1.3, fill: "F4F1EC", line: "DED7CC" });
  s.addText("We had been asking", {
    x: 0.95, y: 1.72, w: 5.15, h: 0.3, margin: 0, fontFace: TEXT, fontSize: 11.5, bold: true, color: MUTED,
  });
  s.addText("“Is it more accurate?”", {
    x: 0.95, y: 2.05, w: 5.15, h: 0.5, margin: 0, fontFace: HEAD, fontSize: 21, bold: true, color: "76839A",
  });

  card(s, { x: 6.85, y: 1.55, w: 5.85, h: 1.3, fill: "E9F5F3", line: "BFE0DA" });
  s.addText("We started asking", {
    x: 7.2, y: 1.72, w: 5.15, h: 0.3, margin: 0, fontFace: TEXT, fontSize: 11.5, bold: true, color: TEAL,
  });
  s.addText("“Does it resist bad labels?”", {
    x: 7.2, y: 2.05, w: 5.15, h: 0.5, margin: 0, fontFace: HEAD, fontSize: 21, bold: true, color: NAVY,
  });

  s.addText(
    "Real datasets contain wrong labels. A big model will happily memorise them, and that is what " +
    "destroys accuracy late in training. Margin methods are supposed to help exactly here — so we built a test for it.",
    { x: 0.6, y: 3.1, w: 12.1, h: 0.7, margin: 0, fontFace: TEXT, fontSize: 15, color: BODY, lineSpacing: 21 }
  );

  const setup = [
    ["40%", "of training labels deliberately corrupted", CHERRY],
    ["100%", "of validation labels kept clean, so we measure real quality", TEAL],
    ["Identical", "corruption for every method — an exact comparison, not a statistical one", "3E6FA8"],
  ];
  setup.forEach(([v, l, col], i) => {
    const y = 4.05 + i * 0.78;
    s.addShape(pres.ShapeType.roundRect, {
      x: 0.6, y, w: 12.1, h: 0.66, rectRadius: 0.1,
      fill: { color: WHITE }, line: { color: CARD_EDGE, width: 1 }, shadow: shadow({ blur: 6, opacity: 0.18 }),
    });
    s.addText(v, {
      x: 0.95, y, w: 1.55, h: 0.66, margin: 0, valign: "middle",
      fontFace: HEAD, fontSize: 20, bold: true, color: col,
    });
    s.addText(l, {
      x: 2.6, y, w: 9.8, h: 0.66, margin: 0, valign: "middle",
      fontFace: TEXT, fontSize: 13.5, color: NAVY,
    });
  });

  s.addText("New measurement: not just accuracy, but how much of the corrupted labels the model actually memorised.", {
    x: 0.6, y: 6.4, w: 12.1, h: 0.34, margin: 0,
    fontFace: TEXT, fontSize: 13, italic: true, bold: true, color: TEAL,
  });

  s.addNotes(
    "[45s   ·   clock 5:30]\n\n" +
    "So we changed the question — from 'is it more accurate' to 'does it resist bad labels'.\n\n" +
    "Every real dataset contains wrong labels, and a model like BERT will cheerfully " +
    "memorise them. That is what wrecks accuracy late in training, and margin methods are " +
    "supposed to be good at exactly this.\n\n" +
    "So: we corrupt 40% of training labels. Validation stays clean, so we always measure " +
    "real quality. And every method sees the IDENTICAL corrupted labels — an exact " +
    "comparison, not a statistical one.\n\n" +
    "We also added the measurement that unlocked the project: how much of the corrupted " +
    "labels did the model actually memorise?"
  );
}

// =====================================================================  9
{
  const s = lightSlide("The new idea: let the gap go NEGATIVE", "Our contribution");

  s.addText(
    "Nothing in the maths says a margin has to be positive. The gap p_t − p_j lives between −1 and +1, " +
    "so δ = −0.4 is a perfectly legal threshold. It means: drop this wrong answer even if it is currently WINNING.",
    { x: 0.6, y: 1.5, w: 12.1, h: 0.75, margin: 0, fontFace: TEXT, fontSize: 15, color: BODY, lineSpacing: 21 }
  );

  card(s, { x: 0.6, y: 2.45, w: 5.85, h: 2.75, fill: "FBEDEE", line: "F0CFD2" });
  circleBadge(s, { x: 0.92, y: 2.68, d: 0.44, fill: CHERRY, label: "✕", size: 14 });
  s.addText("A mislabelled example, positive gap", {
    x: 1.5, y: 2.68, w: 4.7, h: 0.44, margin: 0, valign: "middle",
    fontFace: HEAD, fontSize: 15.5, bold: true, color: NAVY,
  });
  s.addText(
    "The model correctly distrusts the wrong label, so it gives it a low probability.\n\n" +
    "That means no competitor is ever far enough ahead to be dropped — so the model gets pushed " +
    "to fit the wrong label, over and over.",
    { x: 0.92, y: 3.25, w: 5.2, h: 1.35, margin: 0, fontFace: TEXT, fontSize: 12.5, color: BODY, lineSpacing: 17 }
  );
  s.addText("This is exactly how networks memorise noise.", {
    x: 0.92, y: 4.68, w: 5.2, h: 0.35, margin: 0, fontFace: TEXT, fontSize: 12.5, bold: true, color: CHERRY,
  });

  card(s, { x: 6.85, y: 2.45, w: 5.85, h: 2.75, fill: "E9F5F3", line: "BFE0DA" });
  circleBadge(s, { x: 7.17, y: 2.68, d: 0.44, fill: TEAL, label: "✓", size: 14 });
  s.addText("Same example, negative gap", {
    x: 7.75, y: 2.68, w: 4.7, h: 0.44, margin: 0, valign: "middle",
    fontFace: HEAD, fontSize: 15.5, bold: true, color: NAVY,
  });
  s.addText(
    "Now every competitor clears the bar and gets dropped — including the ones that are ahead.\n\n" +
    "Nothing is left to train against, so this example's contribution falls to zero and it " +
    "quietly leaves training.",
    { x: 7.17, y: 3.25, w: 5.2, h: 1.35, margin: 0, fontFace: TEXT, fontSize: 12.5, color: BODY, lineSpacing: 17 }
  );
  s.addText("The suspicious example removes itself.", {
    x: 7.17, y: 4.68, w: 5.2, h: 0.35, margin: 0, fontFace: TEXT, fontSize: 12.5, bold: true, color: TEAL,
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: 5.45, w: 12.1, h: 1.15, rectRadius: 0.12,
    fill: { color: NAVY }, line: { color: NAVY },
  });
  s.addText("Why this matters beyond our project", {
    x: 0.95, y: 5.58, w: 11.4, h: 0.3, margin: 0,
    fontFace: TEXT, fontSize: 11.5, bold: true, color: "6FD8C9",
  });
  s.addText(
    "“Throw away suspicious training examples” is a well-known trick in the noisy-label literature. " +
    "Written this way, it stops being a separate trick — it is just a margin that went negative.",
    { x: 0.95, y: 5.9, w: 11.4, h: 0.6, margin: 0, fontFace: TEXT, fontSize: 13.5, color: WHITE, lineSpacing: 18 }
  );

  s.addNotes(
    "[70s   ·   clock 6:40]\n\n" +
    "This is the idea I'd most like you to take away.\n\n" +
    "A margin doesn't have to be positive. The gap between two probabilities runs from minus " +
    "one to plus one, so a NEGATIVE margin is perfectly legal. It means: drop this wrong " +
    "answer even if it's currently winning.\n\n" +
    "Why that's powerful — left box. Take an example whose label is wrong. The model has seen " +
    "thousands of correct examples, so it correctly distrusts this label and gives it low " +
    "probability. But that means nothing ever gets far enough ahead to be dropped, so the " +
    "model keeps getting pushed to fit the wrong label. That IS memorisation.\n\n" +
    "Right box: with a negative gap, every competitor gets dropped, there's nothing left to " +
    "train against, and the example removes itself from training.\n\n" +
    "And the nice part: 'throw away suspicious examples' is a known trick in this field. " +
    "Written our way it isn't a separate trick any more — it's just a margin that went negative."
  );
}

// ===================================================================== 10
{
  const s = lightSlide("This one worked", "Results, part 2");

  s.addChart(
    pres.ChartType.line,
    [
      { name: "Plain cross-entropy", labels: ["Epoch 1", "Epoch 2", "Epoch 3", "Epoch 4"], values: [3.5, 10.0, 17.8, 33.7] },
      { name: "AS-Softmax", labels: ["Epoch 1", "Epoch 2", "Epoch 3", "Epoch 4"], values: [4.0, 9.8, 18.9, 39.9] },
      { name: "Ours: class-pair gap", labels: ["Epoch 1", "Epoch 2", "Epoch 3", "Epoch 4"], values: [3.6, 8.4, 17.3, 35.2] },
      { name: "Ours: sample gap", labels: ["Epoch 1", "Epoch 2", "Epoch 3", "Epoch 4"], values: [4.0, 10.2, 15.4, 19.3] },
    ],
    {
      x: 0.5, y: 1.5, w: 7.5, h: 4.35,
      chartColors: ["8A93A6", CHERRY, AMBER, TEAL],
      lineSize: 3, lineSmooth: false,
      showLegend: true, legendPos: "b", legendFontFace: TEXT, legendFontSize: 10,
      showTitle: true, title: "How much of the wrong labels did it memorise?  (lower is better)",
      titleFontFace: TEXT, titleFontSize: 12, titleColor: BODY,
      catAxisLabelColor: MUTED, catAxisLabelFontSize: 10, catAxisLabelFontFace: TEXT,
      valAxisLabelColor: MUTED, valAxisLabelFontSize: 10, valAxisLabelFontFace: TEXT,
      valAxisTitle: "% of corrupted examples memorised", showValAxisTitle: true,
      valAxisTitleFontSize: 10, valAxisTitleColor: MUTED, valAxisTitleFontFace: TEXT,
      valGridLine: { color: "E6EAF0", size: 1 },
      catGridLine: { style: "none" },
      valAxisMaxVal: 45, valAxisMinVal: 0,
    }
  );

  const stats = [
    ["4× less", "memorisation of wrong labels", TEAL],
    ["+2.8 pts", "better than cross-entropy at the end of training", TEAL],
    ["Both seeds", "the effect replicated — it is not a lucky run", "3E6FA8"],
  ];
  stats.forEach(([v, l, col], i) => {
    const y = 1.72 + i * 1.42;
    card(s, { x: 8.3, y, w: 4.4, h: 1.2, fill: WHITE });
    s.addText(v, {
      x: 8.6, y: y + 0.15, w: 3.8, h: 0.5, margin: 0,
      fontFace: HEAD, fontSize: 26, bold: true, color: col,
    });
    s.addText(l, {
      x: 8.6, y: y + 0.66, w: 3.8, h: 0.42, margin: 0,
      fontFace: TEXT, fontSize: 12, color: BODY,
    });
  });

  s.addText(
    "Perfect resistance would be a flat line along the bottom. Cross-entropy climbs to 34% and AS-Softmax to 40% — " +
    "our sample gap stops at 19%. Measured a second way, ours ends 3.1 points above a no-memorisation floor where cross-entropy ends 11.5 above it.",
    { x: 0.5, y: 6.1, w: 12.2, h: 0.6, margin: 0, fontFace: TEXT, fontSize: 12.5, italic: true, color: MUTED, lineSpacing: 17 }
  );

  s.addNotes(
    "[60s   ·   clock 7:40]\n\n" +
    "And this worked.\n\n" +
    "The chart is memorisation of the corrupted labels over training — lower is better.\n\n" +
    "Grey is plain cross-entropy, climbing to 34%. Red is AS-Softmax, and notice it's the " +
    "WORST at 40% — I'll come back to that. Amber is our class-pair idea, no better than " +
    "cross-entropy. Teal is the sample gap: it flattens out at 19%.\n\n" +
    "Roughly four times less memorisation, and at the end of training it's 2.8 points more " +
    "accurate than cross-entropy. And it replicated on both random seeds — the two runs gave " +
    "almost identical numbers, so this isn't a lucky seed.\n\n" +
    "One more thing worth flagging: AS-Softmax memorised MORE than plain cross-entropy, on " +
    "both seeds. That contradicts the standard argument for why margin methods should help " +
    "with noisy labels. That's a genuine negative finding and nobody could have seen it by " +
    "looking at accuracy alone."
  );
}

// ===================================================================== 11
{
  const s = lightSlide("Being honest about what we can claim", "Rigour");

  s.addText(
    "With only two runs per method, some of these numbers are solid and some are not. We grade them separately.",
    { x: 0.6, y: 1.52, w: 12.1, h: 0.4, margin: 0, fontFace: TEXT, fontSize: 14.5, color: BODY }
  );

  const rows = [
    ["4× less memorisation of wrong labels", "CONFIRMED", "identical on both runs; the effect is ~20× the measurement's own noise", TEAL],
    ["AS-Softmax memorises more than plain CE", "CONFIRMED", "held on both runs — and it contradicts the textbook explanation", TEAL],
    ["+2.8 pts at the end of training", "CONFIRMED", "both runs positive, though the size varied (+3.7 and +1.8)", TEAL],
    ["Our class-pair idea helps", "REJECTED", "flat across all four testbeds, and not a tuning artefact", CHERRY],
    ["+0.5 pts on best-case accuracy", "NOT ESTABLISHED", "one of the two runs produced the entire gap — we do not claim it", AMBER],
    ["Better calibration", "NO CLAIM", "the difference is an artefact of the method, not a real gain", MUTED],
  ];
  rows.forEach(([claim, badge, why, col], i) => {
    const y = 2.12 + i * 0.76;
    s.addShape(pres.ShapeType.roundRect, {
      x: 0.6, y, w: 12.1, h: 0.64, rectRadius: 0.1,
      fill: { color: WHITE }, line: { color: CARD_EDGE, width: 1 }, shadow: shadow({ blur: 5, opacity: 0.15 }),
    });
    s.addText(claim, {
      x: 0.9, y, w: 4.45, h: 0.64, margin: 0, valign: "middle",
      fontFace: TEXT, fontSize: 13, bold: true, color: NAVY,
    });
    s.addShape(pres.ShapeType.roundRect, {
      x: 5.5, y: y + 0.14, w: 1.85, h: 0.36, rectRadius: 0.08,
      fill: { color: col }, line: { color: col },
    });
    s.addText(badge, {
      x: 5.5, y: y + 0.14, w: 1.85, h: 0.36, align: "center", valign: "middle", margin: 0,
      fontFace: TEXT, fontSize: 9.5, bold: true, color: WHITE,
    });
    s.addText(why, {
      x: 7.55, y, w: 4.85, h: 0.64, margin: 0, valign: "middle",
      fontFace: TEXT, fontSize: 11.5, color: BODY,
    });
  });

  s.addNotes(
    "[45s   ·   clock 8:25]\n\n" +
    "A moment on rigour, because this is where a project like this earns trust or loses it.\n\n" +
    "Two runs per method is not many, so we graded every claim separately.\n\n" +
    "Confirmed: the memorisation result — both runs nearly identical, and the effect is about " +
    "twenty times the measurement's own noise. Rejected: our original class-pair idea.\n\n" +
    "This row matters most. The half-point gain on best-case accuracy clears the 0.3% bar we " +
    "set in advance — but one of the two runs produced the entire gap. So we do NOT claim it. " +
    "It would have been easy to report and move on. It would not have replicated."
  );
}

// ===================================================================== 12
{
  const s = lightSlide("So — is it any good?", "The verdict");

  card(s, { x: 0.6, y: 1.55, w: 3.87, h: 2.35, fill: "E9F5F3", line: "BFE0DA" });
  circleBadge(s, { x: 0.92, y: 1.8, d: 0.46, fill: TEAL, label: "✓", size: 15 });
  s.addText("Yes, as a defence", {
    x: 1.5, y: 1.8, w: 2.7, h: 0.46, margin: 0, valign: "middle",
    fontFace: HEAD, fontSize: 17, bold: true, color: NAVY,
  });
  s.addText(
    "It substantially stops the model memorising wrong labels, the mechanism is understood, " +
    "and the effect replicated.",
    { x: 0.92, y: 2.4, w: 3.25, h: 1.3, margin: 0, fontFace: TEXT, fontSize: 12.5, color: BODY, lineSpacing: 17 }
  );

  card(s, { x: 4.72, y: 1.55, w: 3.87, h: 2.35, fill: "FBEDEE", line: "F0CFD2" });
  circleBadge(s, { x: 5.04, y: 1.8, d: 0.46, fill: CHERRY, label: "✕", size: 15 });
  s.addText("No, as an upgrade", {
    x: 5.62, y: 1.8, w: 2.7, h: 0.46, margin: 0, valign: "middle",
    fontFace: HEAD, fontSize: 17, bold: true, color: NAVY,
  });
  s.addText(
    "It does not make the model more accurate at its best moment. Our original hypothesis " +
    "— smarter gap, better accuracy — was wrong.",
    { x: 5.04, y: 2.4, w: 3.25, h: 1.3, margin: 0, fontFace: TEXT, fontSize: 12.5, color: BODY, lineSpacing: 17 }
  );

  card(s, { x: 8.84, y: 1.55, w: 3.87, h: 2.35, fill: "F0F4FA", line: "CFDBEC" });
  circleBadge(s, { x: 9.16, y: 1.8, d: 0.46, fill: "3E6FA8", label: "!", size: 15 });
  s.addText("It depends", {
    x: 9.74, y: 1.8, w: 2.7, h: 0.46, margin: 0, valign: "middle",
    fontFace: HEAD, fontSize: 17, bold: true, color: NAVY,
  });
  s.addText(
    "The gain only shows up if you cannot stop training early using clean data — " +
    "which is exactly the situation when your labels are noisy.",
    { x: 9.16, y: 2.4, w: 3.25, h: 1.3, margin: 0, fontFace: TEXT, fontSize: 12.5, color: BODY, lineSpacing: 17 }
  );

  s.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: 4.15, w: 12.11, h: 1.05, rectRadius: 0.12,
    fill: { color: NAVY }, line: { color: NAVY },
  });
  s.addText(
    "The honest one-liner: we set out to improve accuracy and failed — but we found a real, " +
    "repeatable defence against bad labels, and we can explain exactly why it works.",
    { x: 0.95, y: 4.15, w: 11.4, h: 1.05, margin: 0, valign: "middle",
      fontFace: HEAD, fontSize: 16, color: WHITE, lineSpacing: 22 }
  );

  s.addText("What this project taught us about doing research", {
    x: 0.6, y: 5.45, w: 12.1, h: 0.32, margin: 0, fontFace: TEXT, fontSize: 12.5, bold: true, color: MUTED,
  });
  const lessons = [
    ["Check the foundation first", "we should have verified the baseline's advantage in week 1, not week 8"],
    ["Measure the mechanism, not just the score", "accuracy was flat for six weeks and explained nothing"],
  ];
  lessons.forEach(([h, d], i) => {
    const x = 0.6 + i * 6.25;
    circleBadge(s, { x, y: 5.85, d: 0.4, fill: "3E6FA8", label: String(i + 1), size: 13 });
    s.addText(h, {
      x: x + 0.54, y: 5.83, w: 5.5, h: 0.28, margin: 0,
      fontFace: TEXT, fontSize: 13, bold: true, color: NAVY,
    });
    s.addText(d, {
      x: x + 0.54, y: 6.11, w: 5.5, h: 0.5, margin: 0,
      fontFace: TEXT, fontSize: 11.5, color: MUTED,
    });
  });

  s.addNotes(
    "[45s   ·   clock 9:10]\n\n" +
    "So, is it any good? Three answers.\n\n" +
    "As a defence against bad labels — yes. Real, replicated, mechanism understood.\n\n" +
    "As a general upgrade — no. Our original hypothesis was wrong and I am not going to " +
    "dress that up.\n\n" +
    "And it depends: the gain only appears if you cannot early-stop on clean data — which is " +
    "precisely the situation when labels are noisy. The condition and the use case coincide.\n\n" +
    "Two lessons: check the foundation first, and measure the mechanism, not just the score."
  );
}

// ===================================================================== 13
{
  const s = lightSlide("What we would do next", "Next steps");

  const items = [
    ["Finish the dial test", "Run the half-strength version. If halving the setting halves the effect, that is strong proof the mechanism is what we think it is.", "≈1 hour of GPU", TEAL],
    ["Compare against the real competition", "We beat cross-entropy and AS-Softmax. We have not compared against the methods actually designed for noisy labels.", "the honest next bar", "3E6FA8"],
    ["Try realistic noise", "Ours is random noise. Real mislabelling is systematic — confusable classes get swapped — and is much harder to detect.", "harder test", AMBER],
    ["Test where margins should shine", "Thousands of classes, or heavily imbalanced data — the regimes where skipping easy answers actually matters.", "beyond our GPU", MUTED],
  ];
  items.forEach(([h, d, tag, col], i) => {
    const x = 0.6 + (i % 2) * 6.25;
    const y = 1.62 + Math.floor(i / 2) * 2.42;
    card(s, { x, y, w: 5.86, h: 2.16, fill: WHITE });
    circleBadge(s, { x: x + 0.3, y: y + 0.26, d: 0.5, fill: col, label: String(i + 1) });
    s.addText(h, {
      x: x + 0.94, y: y + 0.26, w: 4.6, h: 0.5, margin: 0, valign: "middle",
      fontFace: HEAD, fontSize: 16.5, bold: true, color: NAVY,
    });
    s.addText(d, {
      x: x + 0.3, y: y + 0.87, w: 5.26, h: 0.85, margin: 0,
      fontFace: TEXT, fontSize: 12.5, color: BODY, lineSpacing: 17,
    });
    s.addText(tag, {
      x: x + 0.3, y: y + 1.72, w: 5.26, h: 0.3, margin: 0,
      fontFace: TEXT, fontSize: 11, bold: true, italic: true, color: col === MUTED ? MUTED : col,
    });
  });

  s.addText("Everything is reproducible: one command per experiment, results written as machine-readable records.", {
    x: 0.6, y: 6.5, w: 12.1, h: 0.34, margin: 0, align: "center",
    fontFace: TEXT, fontSize: 12.5, italic: true, color: MUTED,
  });

  s.addNotes(
    "[30s   ·   clock 9:40]\n\n" +
    "Four next steps, in order of value per hour.\n\n" +
    "One, about an hour of GPU: run the half-strength version. If halving the setting halves " +
    "the effect, that is strong evidence the mechanism is what we say. We ran out of time.\n\n" +
    "Two, the honest one: we have not compared against methods actually built for noisy " +
    "labels. Until we do, our claim stays narrow.\n\n" +
    "Three, realistic noise — real mislabelling is systematic, and harder to detect.\n\n" +
    "Four, the regimes where margins should shine: thousands of classes, or imbalanced data."
  );
}

// ===================================================================== 14
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  s.addText("Thank you", {
    x: 0.85, y: 1.55, w: 11.6, h: 0.9, margin: 0,
    fontFace: HEAD, fontSize: 42, bold: true, color: WHITE,
  });
  s.addText("Questions welcome — including the sceptical ones.", {
    x: 0.85, y: 2.42, w: 11.6, h: 0.4, margin: 0,
    fontFace: TEXT, fontSize: 16, color: "A9B6CC",
  });

  const takeaways = [
    ["A margin can be negative.", "and if you let it be, discarding suspicious training examples becomes a special case of a margin", "6FD8C9"],
    ["It cut memorisation of wrong labels ~4×.", "replicated on both runs, with the mechanism understood", "6FD8C9"],
    ["It did not improve peak accuracy.", "our original hypothesis was wrong, and we report it that way", "FF8A8F"],
  ];
  takeaways.forEach(([h, d, col], i) => {
    const y = 3.35 + i * 1.02;
    s.addShape(pres.ShapeType.roundRect, {
      x: 0.85, y, w: 11.6, h: 0.88, rectRadius: 0.1,
      fill: { color: NAVY_SOFT }, line: { color: "35496B", width: 1 },
    });
    circleBadge(s, { x: 1.15, y: y + 0.2, d: 0.48, fill: col, label: String(i + 1), labelColor: NAVY });
    s.addText(h, {
      x: 1.8, y: y + 0.11, w: 10.4, h: 0.34, margin: 0,
      fontFace: TEXT, fontSize: 15, bold: true, color: WHITE,
    });
    s.addText(d, {
      x: 1.8, y: y + 0.45, w: 10.4, h: 0.32, margin: 0,
      fontFace: TEXT, fontSize: 12, color: "A9B6CC",
    });
  });

  s.addText("Code, full report and every result table:   FINAL_REPORT.md", {
    x: 0.85, y: 6.65, w: 11.6, h: 0.32, margin: 0,
    fontFace: TEXT, fontSize: 12, color: "7C8AA5",
  });

  s.addNotes(
    "[20s   ·   clock 10:00]\n\n" +
    "Three things to take away.\n\n" +
    "A margin can be negative — and once you allow that, discarding suspicious examples " +
    "becomes a special case of a margin.\n\n" +
    "That cut memorisation of wrong labels about fourfold, replicated.\n\n" +
    "And it did not improve peak accuracy — our hypothesis was wrong and we report it that " +
    "way, because a negative result you can trust beats a positive one you cannot.\n\n" +
    "Happy to take questions."
  );
}

pres.writeFile({ fileName: "GAM-Softmax-Presentation.pptx" }).then((f) => console.log("wrote", f));
