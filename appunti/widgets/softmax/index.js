import katex from 'katex';
import { defineWidget, h, s, svgRoot } from 'texweb/client';
import './style.css';

// il softmax di un vettore di logit: con i cursori si cambiano gli z_i e si vede come l'esponenziale
// e la normalizzazione schiacciano le probabilità verso 0 o 1

export default defineWidget({
  name: 'softmax',
  params: {
    z: { type: 'list', default: [0.6, 1.1, -1.5, 1.2, 3.2, -1.1] },
  },

  mount(el, { z: z0 }) {
    el.classList.add('w-softmax-nlp');
    const z = z0.map(Number), K = z.length;

    const W = 620, H = 250, X0 = 40, slot = (W - 2 * X0) / K, bw = slot * 0.3;
    const ZY = 90, ZS = 18;    // barre dei logit: asse a y = ZY, 18 px per unità
    const PY = 235, PS = 70;   // barre delle probabilità: base a y = PY, 70 px per 1
    const svg = svgRoot(W, H, { 'aria-label': `Softmax di ${K} logit` });
    svg.append(s('line', { x1: X0, y1: ZY, x2: W - X0, y2: ZY, class: 'sx-axis' }), s('line', { x1: X0, y1: PY, x2: W - X0, y2: PY, class: 'sx-axis' }),
      s('text', { x: 4, y: ZY - 40, class: 'sx-head' }, 'z'), s('text', { x: 4, y: PY - 50, class: 'sx-head' }, 'p'));
    const bars = z.map((_, i) => {
      const cx = X0 + i * slot + slot / 2;
      const zb = s('rect', { x: cx - bw - 2, width: bw, class: 'sx-z' });
      const pb = s('rect', { x: cx + 2, width: bw, class: 'sx-p' });
      const zt = s('text', { x: cx - bw / 2 - 2, class: 'sx-val' });
      const pt = s('text', { x: cx + bw / 2 + 2, class: 'sx-val' });
      svg.append(zb, pb, zt, pt);
      return { zb, pb, zt, pt };
    });

    const eq = h('div', { class: 'eq-line' });
    function draw() {
      const e = z.map(Math.exp), sum = e.reduce((a, v) => a + v, 0), p = e.map((v) => v / sum);
      bars.forEach(({ zb, pb, zt, pt }, i) => {
        const hz = Math.abs(z[i]) * ZS;
        zb.setAttribute('y', z[i] >= 0 ? ZY - hz : ZY); zb.setAttribute('height', hz);
        zb.classList.toggle('neg', z[i] < 0);
        zt.setAttribute('y', z[i] >= 0 ? ZY - hz - 4 : ZY + hz + 12); zt.textContent = z[i].toFixed(1);
        pb.setAttribute('y', PY - p[i] * PS); pb.setAttribute('height', p[i] * PS);
        pt.setAttribute('y', PY - p[i] * PS - 4); pt.textContent = p[i] < 0.01 ? p[i].toFixed(3) : p[i].toFixed(2);
      });
      katex.render(`\\mathrm{softmax}(z) = [${p.map((v) => (v < 0.01 ? v.toFixed(4) : v.toFixed(3))).join(',\\ ')}],\\qquad \\textstyle\\sum_j e^{z_j} = ${sum.toFixed(2)}`, eq, { throwOnError: false });
    }

    const sliders = h('div', { class: 'sliders sx-sliders' }, z.map((v, i) => {
      const out = h('output', { class: 'sl-val' }, v.toFixed(1));
      return h('label', { class: 'sl' }, [h('span', { class: 'sl-name' }, `z${'₁₂₃₄₅₆₇₈₉'[i] ?? i + 1}`),
        h('input', { type: 'range', min: -3, max: 4, step: 0.1, value: v, 'aria-label': `z${i + 1}`,
          oninput: (ev) => { z[i] = Number(ev.target.value); out.textContent = z[i].toFixed(1); draw(); } }), out]);
    }));
    el.append(sliders, svg, eq);
    draw();
  },
});
