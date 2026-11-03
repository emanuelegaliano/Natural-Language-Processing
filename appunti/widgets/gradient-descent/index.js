import katex from 'katex';
import { defineWidget, h, s, svgRoot, createPlayer } from 'texweb/client';
import './style.css';

// la discesa del gradiente con un solo peso w (e bias fissato a 0): la loss è la cross-entropy media
// su un piccolo training set, convessa; a ogni passo si calcola la pendenza e ci si sposta nel verso opposto.
// I pulsanti cambiano il learning rate: troppo basso è lento, troppo alto oscilla

const sigma = (z) => 1 / (1 + Math.exp(-z));

export default defineWidget({
  name: 'gradient-descent',
  params: {
    data: { type: 'matrix', default: [[1, 1], [2, 1], [3, 1], [-1, 0], [-2, 0], [2, 0], [-1, 1]] },
    w0: { type: 'number', default: -1.5 },
    etas: { type: 'list', default: [0.3, 1, 2.5, 4] },
    eta: { type: 'number', default: 1 },
    steps: { type: 'integer', default: 8, min: 1, max: 30 },
    interval: { type: 'number', default: 1300, min: 300 },
  },

  mount(el, { data, w0, etas, eta: eta0, steps, interval }) {
    el.classList.add('w-gd');
    const D = data.map(([x, y]) => [Number(x), Number(y)]);
    const L = (w) => D.reduce((a, [x, y]) => a - (y * Math.log(sigma(w * x)) + (1 - y) * Math.log(1 - sigma(w * x))), 0) / D.length;
    const G = (w) => D.reduce((a, [x, y]) => a + (sigma(w * x) - y) * x, 0) / D.length;
    let eta = Number(eta0), path = [];
    const run = () => { path = [Number(w0)]; for (let t = 0; t < steps; t++) path.push(path[t] - eta * G(path[t])); };
    run();

    const W = 620, H = 260, X0 = 50, Y0 = 220, PW = 540, PH = 190, WM = 3.5, LM = 4.5;
    const px = (w) => X0 + ((w + WM) / (2 * WM)) * PW, py = (l) => Y0 - (Math.min(l, LM) / LM) * PH;
    const svg = svgRoot(W, H, { 'aria-label': 'La loss in funzione del peso w e i passi della discesa del gradiente' });
    const curve = Array.from({ length: 141 }, (_, i) => { const w = -WM + i * 0.05; return `${px(w)},${py(L(w))}`; }).join(' ');
    svg.append(
      s('line', { x1: X0, y1: Y0, x2: X0 + PW, y2: Y0, class: 'gd-axis' }),
      s('line', { x1: X0, y1: Y0, x2: X0, y2: Y0 - PH - 5, class: 'gd-axis' }),
      ...[-3, -2, -1, 0, 1, 2, 3].map((w) => s('text', { x: px(w), y: Y0 + 16, class: 'gd-tick' }, String(w))),
      s('text', { x: X0 + PW, y: Y0 - 6, class: 'gd-tick end' }, 'w'),
      s('text', { x: X0 + 6, y: Y0 - PH, class: 'gd-tick start' }, 'loss'),
      s('polyline', { points: curve, class: 'gd-curve' }),
    );
    const trail = s('polyline', { class: 'gd-trail' });
    const tangent = s('line', { class: 'gd-tangent' });
    const dots = s('g');
    const cur = s('circle', { r: 6, class: 'gd-cur' });
    svg.append(trail, tangent, dots, cur);

    const eq = h('div', { class: 'eq-line' });
    const note = h('p', { class: 'widget-note' });

    function draw(n) {
      const w = path[n], g = G(w);
      const vis = path.slice(0, n + 1).map((v) => Math.max(-WM, Math.min(WM, v)));
      trail.setAttribute('points', vis.map((v) => `${px(v)},${py(L(v))}`).join(' '));
      dots.replaceChildren(...vis.slice(0, -1).map((v) => s('circle', { cx: px(v), cy: py(L(v)), r: 3.5, class: 'gd-dot' })));
      const wc = Math.max(-WM, Math.min(WM, w));
      cur.setAttribute('cx', px(wc)); cur.setAttribute('cy', py(L(wc)));
      // la tangente: un segmento di lunghezza fissa lungo la pendenza g
      const dw = 0.9 / Math.sqrt(1 + (g * PH / LM / (PW / (2 * WM))) ** 2);
      tangent.setAttribute('x1', px(wc - dw)); tangent.setAttribute('y1', py(L(wc) - g * dw));
      tangent.setAttribute('x2', px(wc + dw)); tangent.setAttribute('y2', py(L(wc) + g * dw));
      const next = w - eta * g;
      katex.render(`w_{${n + 1}} = w_{${n}} - \\eta\\,\\frac{dL}{dw} = ${w.toFixed(3)} - ${eta} \\cdot (${g.toFixed(3)}) = ${next.toFixed(3)}, \\qquad L(w_{${n}}) = ${L(w).toFixed(3)}`, eq, { throwOnError: false });
      note.textContent = n === 0
        ? `La pendenza in w₀ è ${g < 0 ? 'negativa: per scendere ci spostiamo verso destra' : 'positiva: per scendere ci spostiamo verso sinistra'}.`
        : eta <= 0.3 ? 'Con un learning rate piccolo i passi sono corti: si arriva al minimo, ma lentamente.'
          : eta >= 4 ? 'Con un learning rate troppo alto si salta da una parte all\'altra del minimo senza mai raggiungerlo.'
            : eta >= 2.5 ? 'Il primo passo supera il minimo, poi l\'algoritmo torna indietro e converge.'
              : 'A ogni passo la pendenza diminuisce, e con lei la lunghezza del passo: vicino al minimo ci si muove poco.';
    }

    const player = createPlayer({ steps: steps + 1, onStep: draw, interval });
    const seg = h('div', { class: 'seg' }, etas.map((v) => {
      const b = h('button', { type: 'button', class: `seg-btn${Number(v) === eta ? ' on' : ''}`, onclick: () => {
        eta = Number(v); seg.querySelectorAll('.seg-btn').forEach((x) => x.classList.toggle('on', x === b)); run(); player.go(0); player.play();
      } }, `η = ${v}`);
      return b;
    }));
    el.append(h('div', { class: 'controls' }, [h('span', { class: 'ctl-label' }, 'learning rate'), seg]), svg, eq, note, player.el);
  },
});
