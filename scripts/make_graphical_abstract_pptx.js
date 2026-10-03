// Graphical abstract as one editable PowerPoint slide (Elsevier: 5:2 ratio, Arial,
// left-to-right reading, no title, no empty margins). Values come from values.json,
// computed by scripts/graphical_abstract_values.py.
// Usage: npm install pptxgenjs@3, then  node scripts/make_graphical_abstract_pptx.js
// PNG/PDF are exported from PowerPoint (File > Export); the TIFF is the PNG saved at 300 dpi.
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const FIG = path.join(__dirname, "..", "paper", "figures");
const V = JSON.parse(fs.readFileSync(path.join(FIG, "graphical_abstract_values.json"), "utf8"));
const OUT = process.argv[2] || path.join(FIG, "graphical_abstract.pptx");

const W = 13.333, H = 5.333;                       // 5:2
const BLUE = "1F4E8C", ORANGE = "E08A1E", GREEN = "2E7D4F", GREY = "6B7280", INK = "1D2433",
      RED = "C23B22", BAR_GREY = "9CA3AF";
const FILL = { data: "EAF2FB", method: "FDF6E3", results: "EEF6F0" };
const FONT = "Arial";

const pres = new pptxgen();
pres.defineLayout({ name: "GA_5x2", width: W, height: H });
pres.layout = "GA_5x2";
pres.title = "Graphical abstract";
pres.author = "Bilali Boureima Cissé";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
const s = pres.addSlide();
s.background = { color: "FFFFFF" };

const T = (txt, x, y, w, h, o = {}) => s.addText(txt, {
  x, y, w, h, isTextBox: true, margin: 0, fontFace: FONT, fontSize: 14, color: INK,
  align: "center", valign: "middle", ...o,
});
const panel = (x, w, fill, line, name) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x, y: 0.08, w, h: H - 0.16, rectRadius: 0.12, fill: { color: fill },
  line: { color: line, width: 1.75 }, objectName: name,
});
const arrow = (x, name) => s.addShape(pres.shapes.RIGHT_ARROW, {
  x, y: H / 2 - 0.22, w: 0.42, h: 0.44, fill: { color: BLUE }, line: { color: BLUE, width: 0 }, objectName: name,
});

// ---------------------------------------------------------------- panel 1: data
const P1 = { x: 0.08, w: 4.05 };
panel(P1.x, P1.w, FILL.data, BLUE, "Panel data");
T("Physical PV data", P1.x, 0.22, P1.w, 0.5, { fontSize: 20, bold: true, color: BLUE });
// sun + rays
const sx = 0.62, sy = 1.08;
s.addShape(pres.shapes.OVAL, { x: sx - 0.26, y: sy - 0.26, w: 0.52, h: 0.52, fill: { color: ORANGE },
  line: { color: ORANGE, width: 0 }, objectName: "Sun" });
for (let k = 0; k < 8; k++) {
  const a = k * Math.PI / 4, r0 = 0.34, r1 = 0.48;
  const x0 = sx + r0 * Math.cos(a), y0 = sy + r0 * Math.sin(a), x1 = sx + r1 * Math.cos(a), y1 = sy + r1 * Math.sin(a);
  s.addShape(pres.shapes.LINE, { x: Math.min(x0, x1), y: Math.min(y0, y1), w: Math.abs(x1 - x0) || 0.001,
    h: Math.abs(y1 - y0) || 0.001, line: { color: ORANGE, width: 2.5 },
    flipH: (x1 - x0) * (y1 - y0) < 0, objectName: `Sun ray ${k + 1}` });
}
// PV array
s.addShape(pres.shapes.PARALLELOGRAM, { x: 0.55, y: 1.62, w: 1.25, h: 0.55, fill: { color: "9CC3EA" },
  line: { color: BLUE, width: 1.75 }, objectName: "PV array" });
s.addShape(pres.shapes.LINE, { x: 1.17, y: 2.17, w: 0.001, h: 0.35, line: { color: INK, width: 3 }, objectName: "PV stand" });
// sites
T("Yulara  768 kW", 2.05, 1.05, 2.0, 0.34, { fontSize: 16, bold: true });
T("off-grid, desert", 2.05, 1.38, 2.0, 0.3, { fontSize: 13, color: GREY });
T("NIST  271 kW", 2.05, 1.85, 2.0, 0.34, { fontSize: 16, bold: true });
T("grid-connected, humid", 2.05, 2.18, 2.0, 0.3, { fontSize: 13, color: GREY });
// curtailment
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.35, y: 2.75, w: 3.5, h: 0.85, rectRadius: 0.08,
  fill: { color: "FFFFFF" }, line: { color: RED, width: 2 }, objectName: "Curtailment box" });
T("Curtailed series removed", 0.35, 2.8, 3.5, 0.4, { fontSize: 15, bold: true, color: RED });
T("≈50% of its energy curtailed", 0.35, 3.17, 3.5, 0.36, { fontSize: 13, color: RED });
// weather
s.addShape(pres.shapes.CLOUD, { x: 0.35, y: 4.05, w: 0.95, h: 0.62, fill: { color: "FFFFFF" },
  line: { color: BLUE, width: 1.75 }, objectName: "Cloud" });
T("Archived GFS forecasts", 1.4, 4.0, 2.65, 0.38, { fontSize: 15, bold: true });
T("only runs published in time", 1.4, 4.37, 2.65, 0.32, { fontSize: 12.5, color: GREY });

arrow(4.2, "Arrow 1");

// ---------------------------------------------------------------- panel 2: method
const P2 = { x: 4.7, w: 4.0 };
panel(P2.x, P2.w, FILL.method, ORANGE, "Panel method");
T("Honest selection (AHSE)", P2.x, 0.22, P2.w, 0.5, { fontSize: 20, bold: true, color: "9A5B05" });
const models = ["LightGBM", "XGBoost", "PatchTST", "N-HiTS", "LSTM", "GRU"];
models.forEach((m, k) => {
  const cx = P2.x + 0.2 + (k % 3) * 1.22, cy = 0.9 + Math.floor(k / 3) * 0.48;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y: cy, w: 1.12, h: 0.38, rectRadius: 0.12,
    fill: { color: "FFFFFF" }, line: { color: ORANGE, width: 1.5 }, objectName: `Model ${m}` });
  T(m, cx, cy, 1.12, 0.38, { fontSize: 13 });
});
T("+ GFS inputs for tree models", P2.x, 1.92, P2.w, 0.3, { fontSize: 13, color: GREY });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: P2.x + 0.22, y: 2.35, w: P2.w - 0.44, h: 1.52, rectRadius: 0.1,
  fill: { color: "FFFFFF" }, line: { color: GREEN, width: 2.25 }, objectName: "Selection rule" });
T("per lead-time block: 1 h, 1–6 h, 6–24 h", P2.x + 0.22, 2.43, P2.w - 0.44, 0.32, { fontSize: 13 });
T("combine models only if the\nbootstrap gain is > 0", P2.x + 0.22, 2.78, P2.w - 0.44, 0.7,
  { fontSize: 16, bold: true, color: GREEN });
T("otherwise: best single model", P2.x + 0.22, 3.5, P2.w - 0.44, 0.3, { fontSize: 12.5, color: GREY });
T("Decided on validation data only", P2.x, 4.03, P2.w, 0.36, { fontSize: 15, bold: true });
T("+ out-of-fold conformal 90% intervals", P2.x, 4.4, P2.w, 0.32, { fontSize: 12.5, color: GREY });

arrow(8.77, "Arrow 2");

// ---------------------------------------------------------------- panel 3: results
const P3 = { x: 9.27, w: 3.98 };
panel(P3.x, P3.w, FILL.results, GREEN, "Panel results");
T("Test results, 1–24 h ahead", P3.x, 0.22, P3.w, 0.5, { fontSize: 20, bold: true, color: GREEN });
T(`GFS lowers MAE: −${V.red_yulara}% (Yulara), −${V.red_nist}% (NIST)`, P3.x, 0.75, P3.w, 0.32,
  { fontSize: 13, bold: true, color: BLUE });
// bar chart drawn with native shapes, to scale (locale-independent decimal points)
T("AHSE test MAE (kW)", P3.x, 1.08, P3.w, 0.28, { fontSize: 12, color: GREY });
const base = 3.2, kw2in = 1.8 / 40;                  // 40 kW -> 1.8 in
const groups = [["Yulara", V.yulara_no_gfs, V.yulara_gfs], ["NIST", V.nist_no_gfs, V.nist_gfs]];
groups.forEach(([site, a, b], g) => {
  const gx = P3.x + 0.55 + g * 1.75;
  [[a, BAR_GREY, "without GFS"], [b, BLUE, "with GFS"]].forEach(([v, col, lab], j) => {
    const bx = gx + j * 0.68, bh = v * kw2in;
    s.addShape(pres.shapes.RECTANGLE, { x: bx, y: base - bh, w: 0.62, h: bh, fill: { color: col },
      line: { color: col, width: 0 }, objectName: `Bar ${site} ${lab} (${v.toFixed(1)} kW)` });
    T(v.toFixed(1), bx - 0.1, base - bh - 0.3, 0.82, 0.28, { fontSize: 13 });
  });
  T(site, gx, base + 0.03, 1.3, 0.32, { fontSize: 15, bold: true });
});
s.addShape(pres.shapes.LINE, { x: P3.x + 0.35, y: base, w: P3.w - 0.7, h: 0, line: { color: GREY, width: 1 },
  objectName: "Baseline" });
[[BAR_GREY, "without GFS", 0.55], [BLUE, "with GFS", 2.3]].forEach(([col, lab, dx]) => {
  s.addShape(pres.shapes.RECTANGLE, { x: P3.x + dx, y: 3.67, w: 0.2, h: 0.2, fill: { color: col },
    line: { color: col, width: 0 }, objectName: `Legend ${lab}` });
  T(lab, P3.x + dx + 0.27, 3.62, 1.3, 0.3, { fontSize: 12.5, align: "left" });
});
T(`90% intervals: coverage ${V.cov_lo.toFixed(2)}–${V.cov_hi.toFixed(2)}`, P3.x, 4.03, P3.w, 0.36,
  { fontSize: 15, bold: true });
T("AHSE never significantly worse than best model", P3.x, 4.4, P3.w, 0.32, { fontSize: 12, color: GREY });

pres.writeFile({ fileName: OUT }).then(f => console.log("written", f));
