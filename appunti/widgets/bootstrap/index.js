import katex from 'katex';
import { defineWidget, h, s, svgRoot, createPlayer } from 'texweb/client';
import { bootstrap, accA, accB, delta } from './model.js';
import './style.css';

// il paired bootstrap test: prima alcuni test set virtuali, uno alla volta, poi tutti i b campioni,
// con l'istogramma di delta(x(i)) e il conteggio dei campioni con delta(x(i)) >= 2 delta(x)

const STEPS = 10;

export default defineWidget({
  name: 'bootstrap',
  params: {
    x: { type: 'list', default: ['ab', 'a', 'ab', 'b', 'a', 'b', 'a', 'ab', '-', 'a'] },
    b: { type: 'integer', default: 10000, min: 100, max: 100000 },
    seed: { type: 'integer', default: 7 },
    interval: { type: 'number', default: 1500, min: 300 },
  },

  mount(el, { x, b, seed, interval }) {
    el.classList.add('w-boot');
    const res = bootstrap(x.map(String), b, seed);
    const d0 = res.d0, thr = 2 * d0;

    // una riga di celle: in ogni cella A e B, barrati se hanno sbagliato
    const cell = (c) => h('span', { class: 'bs-cell' }, [
      h('span', { class: c === 'ab' || c === 'a' ? 'ok' : 'ko' }, 'A'),
      h('span', { class: c === 'ab' || c === 'b' ? 'ok' : 'ko' }, 'B'),
    ]);
    const row = (name, xs) => h('div', { class: 'bs-row' }, [h('span', { class: 'bs-name' }, name), ...xs.map(cell),
      h('span', { class: 'bs-num' }, accA(xs).toFixed(2)), h('span', { class: 'bs-num' }, accB(xs).toFixed(2)),
      h('span', { class: 'bs-num d' }, delta(xs).toFixed(2))]);
    const head = h('div', { class: 'bs-row bs-head' }, [h('span', { class: 'bs-name' }, ''),
      ...x.map((_, i) => h('span', { class: 'bs-cell' }, String(i + 1))),
      h('span', { class: 'bs-num' }, 'A%'), h('span', { class: 'bs-num' }, 'B%'), h('span', { class: 'bs-num d' }, 'δ')]);
    const rows = h('div', { class: 'bs-rows' });

    // istogramma di delta su [-1, 1] a passi di 0.1
    const W = 620, H = 190, X0 = 30, Y0 = 160, HW = 560, HH = 130, NB = 21;
    const bw = HW / NB, bx = (d) => X0 + (Math.round(d * 10) + 10) * bw;
    const svg = svgRoot(W, H, { 'aria-label': 'Istogramma delle differenze di accuratezza sui test set virtuali' });
    const bars = Array.from({ length: NB }, (_, k) => {
      const d = (k - 10) / 10;
      const r = s('rect', { x: X0 + k * bw + 2, width: bw - 4, y: Y0, height: 0, class: `bs-bar${d >= thr - 1e-9 ? ' hot' : ''}` });
      svg.append(r);
      if (k % 2 === 0) svg.append(s('text', { x: X0 + k * bw + bw / 2, y: Y0 + 16, class: 'bs-tick' }, d.toFixed(1)));
      return r;
    });
    svg.append(s('line', { x1: X0, y1: Y0, x2: X0 + HW, y2: Y0, class: 'bs-axis' }),
      s('line', { x1: bx(thr), y1: Y0 - HH - 6, x2: bx(thr), y2: Y0, class: 'bs-thr' }),
      s('text', { x: bx(thr) + 4, y: Y0 - HH, class: 'bs-tick start' }, `2δ(x) = ${thr.toFixed(2)}`),
      s('line', { x1: bx(d0) + bw / 2, y1: Y0 - HH - 6, x2: bx(d0) + bw / 2, y2: Y0, class: 'bs-d0' }),
      s('text', { x: bx(d0) + bw / 2 - 4, y: Y0 - HH, class: 'bs-tick end' }, `δ(x) = ${d0.toFixed(2)}`));

    const eq = h('div', { class: 'eq-line' });
    const note = h('p', { class: 'widget-note' });

    function draw(n) {
      const shown = n <= STEPS ? res.samples.slice(0, n) : null;
      rows.replaceChildren(head, row('x', x), ...(shown ? shown.slice(-4).map((xi, j) => row(`x(${n - Math.min(n, 4) + j + 1})`, xi)) : []));
      const ds = n <= STEPS ? res.deltas.slice(0, n) : res.deltas;
      const counts = new Array(NB).fill(0);
      ds.forEach((d) => { counts[Math.round(d * 10) + 10]++; });
      const max = Math.max(1, ...counts);
      bars.forEach((r, k) => { const hh = (counts[k] / max) * HH; r.setAttribute('y', Y0 - hh); r.setAttribute('height', hh); });
      const sCount = ds.filter((d) => d >= thr - 1e-9).length, m = ds.length;
      katex.render(n === 0
        ? `\\delta(x) = ${accA(x).toFixed(2)} - ${accB(x).toFixed(2)} = ${d0.toFixed(2)}`
        : `s = ${sCount} \\text{ su } ${m} \\text{ campioni},\\qquad \\text{p-value}(x) \\approx \\tfrac{s}{b} = \\tfrac{${sCount}}{${m}} = ${(sCount / m).toFixed(3)}`, eq, { throwOnError: false });
      note.textContent = n === 0
        ? 'Il test set vero: A ha un\'accuratezza di 0.70, B di 0.50. La differenza è reale o è un caso?'
        : n <= STEPS
          ? `Ogni test set virtuale si ottiene estraendo 10 documenti da x con reinserimento: in arancione i campioni con δ ≥ 2δ(x).`
          : `Con tutti i ${b} campioni il p-value è ${res.p.toFixed(3)}, molto sopra la soglia di 0.01: con soli 10 documenti non possiamo rifiutare H0.`;
    }

    const player = createPlayer({ steps: STEPS + 2, onStep: draw, interval });
    el.append(rows, svg, eq, note, player.el);
  },
});
