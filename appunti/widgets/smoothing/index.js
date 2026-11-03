import katex from 'katex';
import { defineWidget, h, s, svgRoot } from 'texweb/client';
import './style.css';

// add-k smoothing sulla distribuzione P(w | denied the): con k = 0 è la stima MLE, con k = 1 è Laplace.
// Il cursore su k e la scelta della dimensione del vocabolario mostrano quanta massa passa alle parole mai viste

export default defineWidget({
  name: 'smoothing',
  params: {
    context: { type: 'string', default: 'denied the' },
    words: { type: 'list', default: ['allegations', 'reports', 'claims', 'request', 'attack', 'man', 'outcome'] },
    counts: { type: 'list', default: [3, 2, 1, 1, 0, 0, 0] },
    k: { type: 'number', default: 1, min: 0, max: 2 },
    vocab: { type: 'list', default: [7, 100, 1446] },
  },

  mount(el, { context, words, counts, k: k0, vocab }) {
    el.classList.add('w-smooth');
    const c = counts.map(Number), N = c.reduce((a, b) => a + b, 0);
    let k = k0, V = Number(vocab[0]);

    const W = 640, H = 250, X0 = 40, Y0 = 200, PH = 160, n = words.length;
    const slot = (W - X0 - 150) / n, bw = slot * 0.62;
    const svg = svgRoot(W, H, { 'aria-label': `Probabilità smussate P(w | ${context})` });
    const py = (p) => Y0 - (p / 0.45) * PH;
    svg.append(s('line', { x1: X0, y1: Y0, x2: X0 + slot * n, y2: Y0, class: 'sm-axis' }));
    [0, 0.1, 0.2, 0.3, 0.4].forEach((t) => svg.append(
      s('line', { x1: X0, y1: py(t), x2: X0 + slot * n, y2: py(t), class: 'sm-grid' }),
      s('text', { x: X0 - 6, y: py(t), class: 'sm-tick' }, t.toFixed(1))));
    const bars = words.map((w, i) => {
      const x = X0 + i * slot + (slot - bw) / 2;
      const mle = s('rect', { x, width: bw, class: 'sm-mle' });
      const bar = s('rect', { x, width: bw, class: `sm-bar${c[i] ? '' : ' zero'}` });
      const val = s('text', { x: x + bw / 2, class: 'sm-val' });
      svg.append(mle, bar, val, s('text', { x: x + bw / 2, y: Y0 + 8, class: 'sm-word', transform: `rotate(35 ${x + bw / 2} ${Y0 + 8})` }, w));
      return { mle, bar, val, i };
    });
    // la massa delle parole mai viste: una barra orizzontale che si riempie
    const MX = X0 + slot * n + 30, MW = 100;
    const massFill = s('rect', { x: MX, y: Y0 - PH, width: 26, class: 'sm-mass' });
    const massText = s('text', { x: MX + 13, y: Y0 - PH - 10, class: 'sm-val' });
    svg.append(s('rect', { x: MX, y: Y0 - PH, width: 26, height: PH, class: 'sm-track' }), massFill, massText,
      s('text', { x: MX + 13, y: Y0 + 16, class: 'sm-word mid' }, 'massa a'),
      s('text', { x: MX + 13, y: Y0 + 30, class: 'sm-word mid' }, 'tutti gli zeri'));

    const eq = h('div', { class: 'eq-line' });
    const note = h('p', { class: 'widget-note' });

    function draw() {
      const den = N + k * V;
      const p = c.map((ci) => (ci + k) / den);
      bars.forEach(({ mle, bar, val, i }) => {
        const pm = c[i] / N;
        mle.setAttribute('y', py(pm)); mle.setAttribute('height', Y0 - py(pm));
        bar.setAttribute('y', py(p[i])); bar.setAttribute('height', Y0 - py(p[i]));
        val.setAttribute('y', py(p[i]) - 6); val.textContent = p[i] >= 0.0005 ? p[i].toFixed(3) : p[i].toExponential(1);
      });
      const zeros = V - c.filter((x) => x > 0).length;
      const mass = (zeros * k) / den;
      massFill.setAttribute('y', Y0 - mass * PH); massFill.setAttribute('height', mass * PH);
      massText.textContent = `${(mass * 100).toFixed(1)}%`;
      katex.render(`P_{\\text{add-}k}(w \\mid \\text{${context}}) = \\frac{C(\\text{${context}}\\ w) + ${k.toFixed(2)}}{${N} + ${k.toFixed(2)} \\cdot ${V}}`, eq, { throwOnError: false });
      note.textContent = k === 0
        ? 'Con k = 0 è la stima MLE: le parole mai viste dopo il contesto hanno probabilità 0.'
        : V > 7
          ? `Con un vocabolario di ${V} parole ci sono ${zeros} zeri, e ognuno riceve la sua parte: alle parole viste resta pochissimo.`
          : 'Le barre tratteggiate sono la stima MLE: lo smoothing toglie massa alle parole viste e la dà a quelle mai viste.';
    }

    const kOut = h('output', { class: 'sl-val' }, k.toFixed(2));
    const slider = h('label', { class: 'sl' }, [h('span', { class: 'sl-name' }, 'k'),
      h('input', { type: 'range', min: 0, max: 2, step: 0.01, value: k, 'aria-label': 'k', oninput: (e) => { k = Number(e.target.value); kOut.textContent = k.toFixed(2); draw(); } }),
      kOut]);
    const seg = h('div', { class: 'seg' }, vocab.map((v) => {
      const b = h('button', { type: 'button', class: `seg-btn${Number(v) === V ? ' on' : ''}`, onclick: () => {
        V = Number(v); seg.querySelectorAll('.seg-btn').forEach((x) => x.classList.toggle('on', x === b)); draw();
      } }, `|V| = ${v}`);
      return b;
    }));
    el.append(h('div', { class: 'controls' }, [h('div', { class: 'sliders' }, slider), seg]), svg, eq, note);
    draw();
  },
});
