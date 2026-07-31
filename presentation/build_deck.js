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

// New slides for the five-section structure: agenda, motivation, dividers,
// experimental setup, and the experimental programme. Injected by restructure.py.

// ---------------------------------------------------------------- helpers
const SECTIONS = [
  ["01", "Introduction", "the problem and what we set out to do"],
  ["02", "Background", "softmax, cross-entropy and the paper we build on"],
  ["03", "Methodology", "how we generalise the margin"],
  ["04", "Experimentation", "datasets, protocol and what we ran"],
  ["05", "Results & Demonstration", "what we found, and what we can claim"],
];

function sectionDivider(activeIdx) {
  const s = pres.addSlide();
  s.background = { color: NAVY };
  const [num, title, sub] = SECTIONS[activeIdx];

  s.addText(num, {
    x: 0.85, y: 2.35, w: 2.2, h: 1.5, margin: 0,
    fontFace: HEAD, fontSize: 84, bold: true, color: "6FD8C9",
  });
  s.addText(title, {
    x: 3.0, y: 2.5, w: 9.4, h: 0.85, margin: 0, valign: "middle",
    fontFace: HEAD, fontSize: 40, bold: true, color: WHITE,
  });
  s.addText(sub, {
    x: 3.05, y: 3.35, w: 9.4, h: 0.4, margin: 0,
    fontFace: TEXT, fontSize: 15, color: "A9B6CC",
  });

  // progress rail: the other four sections, dimmed
  SECTIONS.forEach(([n, t], i) => {
    const x = 0.85 + i * 2.42;
    const on = i === activeIdx;
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 5.55, w: 2.2, h: 0.06, rectRadius: 0.03,
      fill: { color: on ? "6FD8C9" : "2E4160" }, line: { color: on ? "6FD8C9" : "2E4160" },
    });
    s.addText(`${n}  ${t}`, {
      x, y: 5.72, w: 2.3, h: 0.3, margin: 0,
      fontFace: TEXT, fontSize: 9.5, bold: on, color: on ? "6FD8C9" : "5A6B87",
    });
  });
  return s;
}

// ================================================================= AGENDA
function agendaSlide() {
  const s = lightSlide("What I will cover", "Agenda");
  const blurbs = [
    "Why a classifier's training objective is asking for the wrong thing",
    "Softmax, cross-entropy, and AS-Softmax — the method we extend",
    "GAM-Softmax: making the margin adapt, and letting it go negative",
    "Four clean testbeds, then one under 40% wrong labels",
    "What failed, what worked, and what we are willing to claim",
  ];
  SECTIONS.forEach(([n, t], i) => {
    const y = 1.62 + i * 1.02;
    s.addShape(pres.ShapeType.roundRect, {
      x: 0.6, y, w: 12.1, h: 0.88, rectRadius: 0.1,
      fill: { color: i === 4 ? "E9F5F3" : WHITE },
      line: { color: i === 4 ? "BFE0DA" : CARD_EDGE, width: 1 },
      shadow: shadow({ blur: 5, opacity: 0.15 }),
    });
    circleBadge(s, { x: 0.92, y: y + 0.19, d: 0.5, fill: i === 4 ? TEAL : NAVY, label: n, size: 14 });
    s.addText(t, {
      x: 1.58, y: y + 0.1, w: 4.0, h: 0.35, margin: 0, valign: "middle",
      fontFace: HEAD, fontSize: 17, bold: true, color: NAVY,
    });
    s.addText(blurbs[i], {
      x: 1.58, y: y + 0.45, w: 10.6, h: 0.32, margin: 0,
      fontFace: TEXT, fontSize: 12.5, color: BODY,
    });
  });
  s.addNotes(
    "[15s · clock 0:38]\n\nFive parts. Background is deliberately from scratch, so if you " +
    "have never seen a loss function you will still follow the result at the end. Most of the " +
    "time goes to the last two sections."
  );
  return s;
}

// ============================================================= MOTIVATION
function motivationSlide() {
  const s = lightSlide("The problem, and what we set out to do", "Introduction");

  card(s, { x: 0.6, y: 1.55, w: 12.1, h: 1.12, fill: "FBEDEE", line: "F0CFD2" });
  s.addText("The observation this project starts from", {
    x: 0.95, y: 1.7, w: 11.4, h: 0.3, margin: 0,
    fontFace: TEXT, fontSize: 11.5, bold: true, color: CHERRY,
  });
  s.addText(
    "We train classifiers to be 100% certain on every example — but at test time we only ever ask whether the right answer came first.",
    { x: 0.95, y: 2.02, w: 11.4, h: 0.5, margin: 0, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY }
  );

  const blocks = [
    ["Research question", "If a fixed margin beats “aim for 100%”, does a margin that adapts beat a fixed one?", TEAL],
    ["Our approach", "Let the margin vary by class pair, by example, and over training — and allow it to go negative.", "3E6FA8"],
    ["What we deliver", "A tested implementation, five testbeds, and one replicated finding — reported with its limits.", AMBER],
  ];
  blocks.forEach(([h, d, col], i) => {
    const x = 0.6 + i * 4.06;
    card(s, { x, y: 2.95, w: 3.72, h: 2.15, fill: WHITE });
    circleBadge(s, { x: x + 0.3, y: 3.18, d: 0.44, fill: col, label: String(i + 1), size: 14 });
    s.addText(h, {
      x: x + 0.88, y: 3.18, w: 2.7, h: 0.44, margin: 0, valign: "middle",
      fontFace: HEAD, fontSize: 15.5, bold: true, color: NAVY,
    });
    s.addText(d, {
      x: x + 0.3, y: 3.78, w: 3.12, h: 1.15, margin: 0,
      fontFace: TEXT, fontSize: 12.5, color: BODY, lineSpacing: 17,
    });
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: 5.4, w: 12.1, h: 1.1, rectRadius: 0.12,
    fill: { color: NAVY }, line: { color: NAVY },
  });
  s.addText("Where this ends up, so you know where I am going", {
    x: 0.95, y: 5.52, w: 11.4, h: 0.3, margin: 0,
    fontFace: TEXT, fontSize: 11.5, bold: true, color: "6FD8C9",
  });
  s.addText(
    "The accuracy hypothesis failed. The same idea turned out to be a strong defence against wrong labels — which is a different, and smaller, claim.",
    { x: 0.95, y: 5.84, w: 11.4, h: 0.5, margin: 0, fontFace: TEXT, fontSize: 13.5, color: WHITE }
  );

  s.addNotes(
    "[32s · clock 1:10]\n\nThe observation this starts from: we train models to be totally " +
    "certain on every example, but at test time we only check whether the right answer came " +
    "first. Those are not the same requirement.\n\nHence the question: if a fixed margin " +
    "beats aiming for 100%, does an adaptive margin beat a fixed one?\n\nI will tell you the " +
    "ending now, so you judge the evidence rather than wait for a twist. The accuracy " +
    "hypothesis failed. The same idea turned out to be a strong defence against wrong labels."
  );
  return s;
}

// ======================================================= EXPERIMENTAL SETUP
function setupSlide() {
  const s = lightSlide("Experimental setup", "Experimentation");

  const spec = [
    ["Model", "BERT-base, 4 epochs, batch 16, learning rate 2e-5"],
    ["Datasets", "SST-5 — 5 classes, film reviews   ·   20 Newsgroups — 20 classes, forum posts"],
    ["Baselines", "plain cross-entropy   ·   AS-Softmax (the published method we extend)"],
    ["Repeats", "2–3 random seeds per method, so we can see the noise floor"],
    ["Compute", "one 6 GB laptop GPU — about 30 minutes per run"],
  ];
  spec.forEach(([k, v], i) => {
    const y = 1.6 + i * 0.66;
    s.addShape(pres.ShapeType.roundRect, {
      x: 0.6, y, w: 12.1, h: 0.56, rectRadius: 0.09,
      fill: { color: i % 2 ? WHITE : CARD }, line: { color: CARD_EDGE, width: 1 },
    });
    s.addText(k, {
      x: 0.95, y, w: 2.1, h: 0.56, margin: 0, valign: "middle",
      fontFace: TEXT, fontSize: 12.5, bold: true, color: MUTED,
    });
    s.addText(v, {
      x: 3.15, y, w: 9.3, h: 0.56, margin: 0, valign: "middle",
      fontFace: TEXT, fontSize: 13, color: NAVY,
    });
  });

  card(s, { x: 0.6, y: 5.15, w: 5.85, h: 1.42, fill: "E9F5F3", line: "BFE0DA" });
  s.addText("Success criterion, fixed before we looked", {
    x: 0.92, y: 5.32, w: 5.2, h: 0.3, margin: 0,
    fontFace: TEXT, fontSize: 11.5, bold: true, color: TEAL,
  });
  s.addText("Beat AS-Softmax by ≥ 0.3% accuracy, averaged over seeds", {
    x: 0.92, y: 5.66, w: 5.2, h: 0.7, margin: 0,
    fontFace: TEXT, fontSize: 13.5, color: NAVY, lineSpacing: 18,
  });

  card(s, { x: 6.85, y: 5.15, w: 5.85, h: 1.42, fill: WHITE });
  s.addText("What we measured", {
    x: 7.17, y: 5.32, w: 5.2, h: 0.3, margin: 0,
    fontFace: TEXT, fontSize: 11.5, bold: true, color: MUTED,
  });
  s.addText("Accuracy at its best and its final epoch, calibration, and — critically — memorisation of wrong labels",
    { x: 7.17, y: 5.66, w: 5.2, h: 0.75, margin: 0, fontFace: TEXT, fontSize: 12.5, color: NAVY, lineSpacing: 17 });

  s.addNotes(
    "[32s · clock 5:37]\n\nBERT-base on two text datasets — SST-5 with five classes, 20 " +
    "Newsgroups with twenty. Compared against plain cross-entropy and AS-Softmax, two to " +
    "three seeds each, which is what tells us the noise floor. All on one 6 GB laptop " +
    "GPU.\n\nTwo things for the record: the success bar was fixed before we saw any results, " +
    "and we measured more than accuracy — including memorisation, which is what eventually " +
    "explained everything."
  );
  return s;
}

// ==================================================== EXPERIMENTAL PROGRAMME
function programmeSlide() {
  const s = lightSlide("What we ran, in two phases", "Experimentation");

  card(s, { x: 0.6, y: 1.55, w: 5.85, h: 2.5, fill: WHITE });
  circleBadge(s, { x: 0.92, y: 1.8, d: 0.5, fill: "3E6FA8", label: "1" });
  s.addText("Phase 1 — clean labels", {
    x: 1.56, y: 1.8, w: 4.6, h: 0.5, margin: 0, valign: "middle",
    fontFace: HEAD, fontSize: 17, bold: true, color: NAVY,
  });
  s.addText(
    [
      { text: "SST-5, class-pair margin, 3 seeds", options: { bullet: true, breakLine: true } },
      { text: "SST-5 tuning screen — was it under-tuned?", options: { bullet: true, breakLine: true } },
      { text: "20 Newsgroups, class-pair margin", options: { bullet: true } },
    ],
    { x: 0.95, y: 2.45, w: 5.2, h: 1.4, margin: 0, fontFace: TEXT, fontSize: 12.5, color: BODY, paraSpaceAfter: 7 }
  );

  card(s, { x: 6.85, y: 1.55, w: 5.85, h: 2.5, fill: "E9F5F3", line: "BFE0DA" });
  circleBadge(s, { x: 7.17, y: 1.8, d: 0.5, fill: TEAL, label: "2" });
  s.addText("Phase 2 — 40% wrong labels", {
    x: 7.81, y: 1.8, w: 4.6, h: 0.5, margin: 0, valign: "middle",
    fontFace: HEAD, fontSize: 17, bold: true, color: NAVY,
  });
  s.addText(
    "Added when Phase 1 came back flat. Real datasets contain wrong labels, a large model will " +
    "memorise them, and margin methods are supposed to help exactly there.",
    { x: 7.2, y: 2.45, w: 5.2, h: 1.4, margin: 0, fontFace: TEXT, fontSize: 12.5, color: BODY, lineSpacing: 18 }
  );

  s.addText("How the noisy testbed is built", {
    x: 0.6, y: 4.28, w: 12.1, h: 0.32, margin: 0,
    fontFace: TEXT, fontSize: 12.5, bold: true, color: MUTED,
  });
  const setup = [
    ["40%", "of training labels deliberately corrupted", CHERRY],
    ["100%", "of validation labels kept clean, so we always measure real quality", TEAL],
    ["Identical", "corruption for every method — an exact comparison, not a statistical one", "3E6FA8"],
  ];
  setup.forEach(([v, l, col], i) => {
    const y = 4.7 + i * 0.66;
    s.addShape(pres.ShapeType.roundRect, {
      x: 0.6, y, w: 12.1, h: 0.56, rectRadius: 0.09,
      fill: { color: WHITE }, line: { color: CARD_EDGE, width: 1 }, shadow: shadow({ blur: 5, opacity: 0.15 }),
    });
    s.addText(v, {
      x: 0.95, y, w: 1.6, h: 0.56, margin: 0, valign: "middle",
      fontFace: HEAD, fontSize: 18, bold: true, color: col,
    });
    s.addText(l, {
      x: 2.65, y, w: 9.7, h: 0.56, margin: 0, valign: "middle",
      fontFace: TEXT, fontSize: 13, color: NAVY,
    });
  });

  s.addNotes(
    "[35s · clock 6:12]\n\nTwo phases. Phase one, clean labels: three experiments on the " +
    "class-pair margin, including a tuning screen to check we had not simply under-tuned " +
    "it.\n\nPhase two we added when phase one came back flat — 40% wrong labels. Every real " +
    "dataset has some, a big model will memorise them, and margin methods are supposed to be " +
    "good exactly there.\n\nTwo details make it trustworthy: validation labels stay clean, " +
    "and every method sees the identical corrupted labels — an exact comparison, not a " +
    "statistical one."
  );
  return s;
}

// ---------------------------------------------- original slide 1
function orig1() {
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
    "[20s · clock 0:20]\n\nA research project on classification loss functions.\n\nThere is a " +
    "published method that replaces the usual training objective with a 'margin' rule. We " +
    "tried to make that margin smarter. I will tell you what we tried, what failed, and the " +
    "one thing that worked."
  );
}

// ---------------------------------------------- original slide 2
function orig2() {
  const s = lightSlide("A classifier is a student taking a multiple-choice test", "Background · the basics");

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
    "[30s · clock 1:43]\n\nA classifier is a student answering multiple-choice questions. It " +
    "scores every option, and softmax squashes those scores into percentages summing to " +
    "100.\n\nHere it reads a bad review and puts 55% on 'very negative'. It is right — " +
    "because that answer is AHEAD of the others.\n\nBeing right only requires being ahead. It " +
    "does not require 100%. That gap is where this project lives."
  );
}

// ---------------------------------------------- original slide 3
function orig3() {
  const s = lightSlide("But normal training demands 100% — forever", "Background · the problem");

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
    "[35s · clock 2:18]\n\nStandard training — cross-entropy — is never satisfied.\n\nLeft: " +
    "the model is already right, 55% on the correct answer. Test passed.\n\nRight: " +
    "cross-entropy keeps pushing to 99%. That extra work buys nothing at test time, because " +
    "the test only asked whether it was ahead.\n\nThree costs: wasted effort; overfitting, " +
    "because reaching 99% on hard examples means memorising them; and the loss stops " +
    "correlating with accuracy."
  );
}

// ---------------------------------------------- original slide 4
function orig4() {
  const s = lightSlide("The paper we build on: AS-Softmax (Lv et al., 2023)", "Background · prior work");

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
    "[42s · clock 3:00]\n\nThis is the paper we build on — AS-Softmax, 2023.\n\nTheir fix: do " +
    "not demand 100%, demand a gap. Once the right answer beats a wrong one by delta, drop " +
    "that wrong answer and stop pushing.\n\nThey reported better accuracy, better " +
    "calibration, less overfitting, about 1.2x faster training — and this striking number: " +
    "the correlation between loss and validation accuracy went from roughly zero to minus " +
    "0.95.\n\nAnd here is the door they leave open, our starting point: delta is ONE number, " +
    "for every class pair, every example, every moment of training."
  );
}

// ---------------------------------------------- original slide 5
function orig5() {
  const s = lightSlide("Our idea: one number for everything is too crude", "Methodology · our approach");

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
    "[35s · clock 3:38]\n\nOur idea: if one gap is good, a gap that ADAPTS should be better. " +
    "Three directions.\n\nClass-pair: cat versus dog is harder than cat versus aeroplane, so " +
    "they should not share a gap. Sample: individual examples differ — some ambiguous, some " +
    "just wrong. Time: early on the model is guessing, so start loose and tighten.\n\nWe test " +
    "these one at a time, so any gain traces to a specific cause."
  );
}

// ---------------------------------------------- original slide 6
function orig6() {
  const s = lightSlide("What we implemented", "Methodology · implementation");

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
  s.addText("The three variants we built", {
    x: 7.25, y: 4.1, w: 5.1, h: 0.35, margin: 0, fontFace: HEAD, fontSize: 18, bold: true, color: NAVY,
  });
  const variants = [
    ["M3", "class-pair", "δ from a learned low-rank matrix over class pairs", TEAL],
    ["M4", "sample", "δ from batch-relative confidence — may go negative", CHERRY],
    ["M6", "combined", "M4 modulating M3, so δ varies on all three axes", "3E6FA8"],
  ];
  variants.forEach(([tag, name, desc, col], i) => {
    const y = 4.62 + i * 0.68;
    s.addShape(pres.ShapeType.roundRect, {
      x: 7.25, y, w: 0.72, h: 0.34, rectRadius: 0.07,
      fill: { color: col }, line: { color: col },
    });
    s.addText(tag, {
      x: 7.25, y, w: 0.72, h: 0.34, align: "center", valign: "middle", margin: 0,
      fontFace: TEXT, fontSize: 11, bold: true, color: WHITE,
    });
    s.addText(name, {
      x: 8.08, y, w: 1.5, h: 0.34, margin: 0, valign: "middle",
      fontFace: TEXT, fontSize: 12.5, bold: true, color: NAVY,
    });
    s.addText(desc, {
      x: 7.25, y: y + 0.32, w: 5.1, h: 0.3, margin: 0,
      fontFace: TEXT, fontSize: 11, color: BODY,
    });
  });

  s.addNotes(
    "[22s · clock 5:02]\n\nWhat we implemented. Nine loss functions, seventy passing tests, " +
    "three margin variants.\n\nOne thing worth noting: a test proves our loss becomes EXACTLY " +
    "AS-Softmax when the gap is held constant — so we always compare a strict generalisation, " +
    "not a different loss."
  );
}

// ---------------------------------------------- original slide 7
function orig7() {
  const s = darkSlide("Then we ran it. It did not work.", "Results · what failed");

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
    "[45s · clock 7:00]\n\nIt did not work, and I want to be straight about that.\n\nFour " +
    "testbeds. Our class-pair gap never beat AS-Softmax by more than the noise between two " +
    "runs of the same method. We checked whether we had under-tuned it — we let the gap learn " +
    "harder and it got monotonically WORSE.\n\nThe diagnosis at the bottom is the real " +
    "finding. The problem was not our idea, it was underneath us: AS-Softmax itself did not " +
    "reliably beat plain cross-entropy in our hands. We were generalising an advantage that " +
    "was not there.\n\nThat is where many projects quietly stop. We changed the question."
  );
}

// ---------------------------------------------- original slide 9
function orig9() {
  const s = lightSlide("The new idea: let the gap go NEGATIVE", "Methodology · the key idea");

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
    "[62s · clock 4:40]\n\nThis is the idea I would most like you to take away.\n\nA margin " +
    "does not have to be positive. The gap between two probabilities runs from minus one to " +
    "plus one, so a NEGATIVE margin is legal. It means: drop this wrong answer even if it is " +
    "currently winning.\n\nWhy that is powerful — left box. Take an example whose label is " +
    "wrong. The model has seen thousands of correct examples, so it correctly distrusts this " +
    "label and gives it low probability. But that means nothing ever gets far enough ahead to " +
    "be dropped, so the model keeps being pushed to fit the wrong label. That IS " +
    "memorisation.\n\nRight box: with a negative gap every competitor is dropped, nothing is " +
    "left to train against, and the example removes itself.\n\nAnd 'throw away suspicious " +
    "examples' is a known trick in this field. Written our way it is not a separate trick — " +
    "it is a margin that went negative."
  );
}

// ---------------------------------------------- original slide 10
function orig10() {
  const s = lightSlide("This one worked", "Results · what worked");

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
    "[57s · clock 7:57]\n\nAnd this worked. The chart is memorisation of corrupted labels " +
    "over training — lower is better.\n\nGrey is plain cross-entropy, climbing to 34%. Red is " +
    "AS-Softmax, and notice it is the WORST at 40%. Amber is our class-pair idea, no better " +
    "than cross-entropy. Teal is the sample gap: it flattens at 19%.\n\nRoughly four times " +
    "less memorisation, and at the end of training 2.8 points more accurate than " +
    "cross-entropy — replicated on both seeds.\n\nOne more thing: AS-Softmax memorised MORE " +
    "than plain cross-entropy, on both seeds. That contradicts the standard argument for " +
    "margin methods under noisy labels, and nobody could see it by looking at accuracy alone."
  );
}

// ---------------------------------------------- original slide 11
function orig11() {
  const s = lightSlide("Being honest about what we can claim", "Results · rigour");

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
    "[40s · clock 8:37]\n\nA moment on rigour, because this is where a project earns trust or " +
    "loses it.\n\nTwo runs per method is not many, so we graded every claim " +
    "separately.\n\nConfirmed: the memorisation result — both runs nearly identical, effect " +
    "about twenty times the measurement's own noise. Rejected: our original class-pair " +
    "idea.\n\nThis row matters most. The half-point gain on best-case accuracy clears the bar " +
    "we set in advance — but one of the two runs produced the entire gap. So we do NOT claim " +
    "it."
  );
}

// ---------------------------------------------- original slide 12
function orig12() {
  const s = lightSlide("So — is it any good?", "Results · verdict");

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
    "[35s · clock 9:12]\n\nIs it any good? Three answers.\n\nAs a defence against bad labels " +
    "— yes. Real, replicated, mechanism understood.\n\nAs a general upgrade — no. Our " +
    "hypothesis was wrong and I am not going to dress that up.\n\nAnd it depends: the gain " +
    "only appears if you cannot early-stop on clean data, which is precisely the situation " +
    "when labels are noisy."
  );
}

// ---------------------------------------------- original slide 13
function orig13() {
  const s = lightSlide("What we would do next", "Results · next steps");

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
    "[26s · clock 9:38]\n\nFour next steps. One, an hour of GPU: run the half-strength " +
    "version — if halving the setting halves the effect, that is strong evidence for the " +
    "mechanism. Two, compare against methods actually built for noisy labels. Three, " +
    "realistic noise. Four, thousands of classes, where margins should really matter."
  );
}

// ---------------------------------------------- original slide 14
function orig14() {
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
    "[22s · clock 10:00]\n\nThree takeaways. A margin can be negative — and then discarding " +
    "suspicious examples becomes a special case of a margin. That cut memorisation about " +
    "fourfold. And it did not improve peak accuracy — our hypothesis was wrong and we report " +
    "it that way.\n\nHappy to take questions."
  );
}

// =====================================================================
//  FINAL SLIDE ORDER — Introduction · Background · Methodology ·
//                      Experimentation · Results & Demonstration
// =====================================================================
orig1();             // title

sectionDivider(0);   // 01 INTRODUCTION
agendaSlide();       // what I will cover
motivationSlide();   // problem, question, contributions

sectionDivider(1);   // 02 BACKGROUND
orig2();             // softmax: a classifier as a multiple-choice test
orig3();             // cross-entropy demands 100% forever
orig4();             // AS-Softmax (Lv et al. 2023)

sectionDivider(2);   // 03 METHODOLOGY
orig5();             // the three axes
orig9();             // the key idea: a negative margin
orig6();             // implementation + the three variants

sectionDivider(3);   // 04 EXPERIMENTATION
setupSlide();        // model, data, baselines, criterion
programmeSlide();    // two phases; the noisy-label protocol

sectionDivider(4);   // 05 RESULTS & DEMONSTRATION
orig7();             // what failed
orig10();            // what worked
orig11();            // what we can and cannot claim
orig12();            // verdict
orig13();            // next steps
orig14();            // close

pres.writeFile({ fileName: "GAM-Softmax-Presentation.pptx" }).then((f) => console.log("wrote", f));
