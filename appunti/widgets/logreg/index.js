import katex from 'katex';
import { defineWidget, h, s, svgRoot } from 'texweb/client';
import './style.css';

// la regressione logistica sull'esempio della recensione: con i cursori si cambiano le feature,
// si ricalcola z = w·x + b, il punto si sposta sulla sigmoide e si aggiornano decisione e cross-entropy loss

const sigma = (z) => 1 / (1 + Math.exp(-z));

export default defineWidget({
  name: 'logreg',
  params: {
    x: { type: 'list', default: [3, 2, 1, 3, 0, 4.19] },
    w: { type: 'list', default: [2.5, -5.0, -1.2, 0.5, 2.0, 0.7] },
    b: { type: 'number', default: 0.1 },
    names: { type: 'list', default: ['lessico positivo', 'lessico negativo', '"no" nel doc', 'pronomi 1ª e 2ª pers.', '"!" nel doc', 'ln(parole)'] },
    max: { type: 'list', default: [8, 8, 1, 8, 1, 6] },
  },

  mount(el, { x: x0, w, b, names, max }) {
    el.classList.add('w-logreg');
    const x = x0.map(Number);
    let y = 1;

    // la sigmoide su [-8, 8]
    const W = 620, H = 230, X0 = 50, Y0 = 200, PW = 540, PH = 170, ZM = 8;
    const px = (z) => X0 + ((Math.max(-ZM, Math.min(ZM, z)) + ZM) / (2 * ZM)) * PW, py = (p) => Y0 - p * PH;
    const svg = svgRoot(W, H, { 'aria-label': 'La sigmoide e il punto z = w·x + b' });
    const pts = Array.from({ length: 161 }, (_, i) => { const z = -ZM + i * 0.1; return `${px(z)},${py(sigma(z))}`; }).join(' ');
    svg.append(
      s('line', { x1: X0, y1: Y0, x2: X0 + PW, y2: Y0, class: 'lr-axis' }),
      s('line', { x1: px(0), y1: Y0, x2: px(0), y2: Y0 - PH - 10, class: 'lr-axis' }),
      s('line', { x1: X0, y1: py(0.5), x2: X0 + PW, y2: py(0.5), class: 'lr-half' }),
      s('text', { x: X0 + PW, y: py(0.5) - 6, class: 'lr-tick end' }, 'decision boundary 0.5'),
      ...[-8, -4, 0, 4, 8].map((z) => s('text', { x: px(z), y: Y0 + 16, class: 'lr-tick' }, String(z))),
      ...[0, 0.5, 1].map((p) => s('text', { x: px(0) - 6, y: py(p), class: 'lr-tick end mid' }, String(p))),
      s('text', { x: X0 + PW, y: Y0 - 8, class: 'lr-tick end' }, 'z'),
      s('polyline', { points: pts, class: 'lr-curve' }),
    );
    const guideV = s('line', { class: 'lr-guide' }), guideH = s('line', { class: 'lr-guide' });
    const dot = s('circle', { r: 6, class: 'lr-dot' });
    const dotLab = s('text', { class: 'lr-dotlab' });
    svg.append(guideV, guideH, dot, dotLab);

    const eqZ = h('div', { class: 'eq-line' });
    const eqP = h('div', { class: 'eq-line' });
    const eqL = h('div', { class: 'eq-line' });
    const num = (v) => (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : v.toFixed(2));

    function draw() {
      const z = w.reduce((a, wi, i) => a + Number(wi) * x[i], Number(b)), p = sigma(z);
      dot.setAttribute('cx', px(z)); dot.setAttribute('cy', py(p));
      dot.classList.toggle('neg', p <= 0.5);
      guideV.setAttribute('x1', px(z)); guideV.setAttribute('x2', px(z)); guideV.setAttribute('y1', Y0); guideV.setAttribute('y2', py(p));
      guideH.setAttribute('x1', px(0)); guideH.setAttribute('x2', px(z)); guideH.setAttribute('y1', py(p)); guideH.setAttribute('y2', py(p));
      dotLab.setAttribute('x', px(z) + (z > 4 ? -10 : 10)); dotLab.setAttribute('y', py(p) + (p > 0.85 ? 18 : -10));
      dotLab.classList.toggle('end', z > 4);
      dotLab.textContent = `σ(${z.toFixed(3)}) = ${p.toFixed(2)}`;
      katex.render(`z = \\mathbf{w}\\cdot\\mathbf{x} + b = [${w.map(num).join(',\\,')}]\\cdot[${x.map(num).join(',\\,')}] + ${num(Number(b))} = ${z.toFixed(3)}`, eqZ, { throwOnError: false });
      katex.render(`P(+ \\mid x) = \\sigma(z) = ${p.toFixed(2)},\\qquad P(- \\mid x) = ${(1 - p).toFixed(2)},\\qquad \\hat y = ${p > 0.5 ? 1 : 0}`, eqP, { throwOnError: false });
      const loss = -Math.log(y === 1 ? p : 1 - p);
      katex.render(`y = ${y}:\\quad L_{CE} = -\\log ${y === 1 ? '\\sigma(z)' : '(1 - \\sigma(z))'} = -\\log(${(y === 1 ? p : 1 - p).toFixed(2)}) = ${loss.toFixed(2)}`, eqL, { throwOnError: false });
    }

    const sliders = h('div', { class: 'sliders lr-sliders' }, x.map((v, i) => {
      const out = h('output', { class: 'sl-val' }, num(v));
      const binary = Number(max[i]) === 1;
      return h('label', { class: 'sl' }, [
        h('span', { class: 'sl-name' }, `x${i + 1}`),
        h('input', { type: 'range', min: 0, max: Number(max[i]), step: binary ? 1 : i === 5 ? 0.01 : 1, value: v, 'aria-label': `x${i + 1}: ${names[i]}`,
          oninput: (e) => { x[i] = Number(e.target.value); out.textContent = num(x[i]); draw(); } }),
        out,
        h('span', { class: 'lr-feat' }, `${names[i]} (w = ${num(Number(w[i]))})`),
      ]);
    }));
    const seg = h('div', { class: 'seg' }, [1, 0].map((v) => {
      const btn = h('button', { type: 'button', class: `seg-btn${v === y ? ' on' : ''}`, onclick: () => {
        y = v; seg.querySelectorAll('.seg-btn').forEach((bb) => bb.classList.toggle('on', bb === btn)); draw();
      } }, v ? 'classe vera: positiva (y = 1)' : 'classe vera: negativa (y = 0)');
      return btn;
    }));
    el.append(sliders, svg, eqZ, eqP, h('div', { class: 'controls' }, seg), eqL);
    draw();
  },
});
