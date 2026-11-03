import katex from 'katex';
import { defineWidget, h, s, svgRoot, createPlayer } from 'texweb/client';
import './style.css';

// il metodo di visualizzazione di Shannon con un modello a bigrammi stimato (MLE) su un mini corpus:
// a ogni passo la distribuzione P(w | parola precedente) copre l'intervallo [0, 1], si estrae un numero
// a caso e si sceglie la parola il cui intervallo lo contiene, finché non esce </s>

const BOS = '<s>', EOS = '</s>';

// conteggi dei bigrammi: per ogni parola, le parole che la seguono nell'ordine in cui compaiono nel corpus
function bigrams(sentences) {
  const next = new Map();
  for (const sent of sentences) {
    const w = [BOS, ...String(sent).trim().split(/\s+/), EOS];
    for (let i = 0; i + 1 < w.length; i++) {
      if (!next.has(w[i])) next.set(w[i], new Map());
      const m = next.get(w[i]);
      m.set(w[i + 1], (m.get(w[i + 1]) ?? 0) + 1);
    }
  }
  // distribuzioni: [{ w, c, p, lo, hi }] con gli intervalli cumulati
  const dist = new Map();
  for (const [prev, m] of next) {
    const tot = [...m.values()].reduce((a, b) => a + b, 0);
    let acc = 0;
    dist.set(prev, [...m].map(([w, c]) => { const lo = acc; acc += c / tot; return { w, c, tot, p: c / tot, lo, hi: acc }; }));
  }
  return dist;
}

// una frase: la sequenza di estrazioni, ognuna con il contesto, il numero estratto e la parola scelta
function sample(dist, rand, maxLen = 14) {
  const steps = [];
  let prev = BOS;
  while (steps.length < maxLen) {
    const d = dist.get(prev), r = rand(steps.length);
    const pick = d.find((x) => r < x.hi) ?? d[d.length - 1];
    steps.push({ prev, r, d, pick });
    if (pick.w === EOS) break;
    prev = pick.w;
  }
  return steps;
}

const frac = (x) => `\\tfrac{${x.c}}{${x.tot}}`;

export default defineWidget({
  name: 'shannon',
  params: {
    corpus: { type: 'list', default: ['I am Sam', 'Sam I am', 'I do not like green eggs and ham'] },
    r: { type: 'list', default: [0.41, 0.18, 0.3, 0.22] },
    interval: { type: 'number', default: 1800, min: 400 },
  },

  mount(el, { corpus, r: r0, interval }) {
    el.classList.add('w-shannon');
    const dist = bigrams(corpus);
    // la prima frase usa i numeri della figura del PDF, le successive numeri a caso
    let steps = sample(dist, (i) => (i < r0.length ? Number(r0[i]) : Math.random()));

    const W = 620, X0 = 60, BW = 520, BY = 46, BH = 34;
    const svg = svgRoot(W, 110, { 'aria-label': 'Estrazione della parola successiva da un modello a bigrammi' });
    const ctx = s('text', { x: X0, y: 24, class: 'sh-ctx' });
    const segs = s('g');
    const needle = s('g', { class: 'sh-needle' }, [
      s('line', { x1: 0, y1: BY - 8, x2: 0, y2: BY + BH + 8 }),
      s('text', { x: 0, y: BY + BH + 24, class: 'sh-r' }),
    ]);
    svg.append(ctx, segs, needle,
      s('text', { x: X0 - 8, y: BY + BH / 2, class: 'sh-tick end' }, '0'),
      s('text', { x: X0 + BW + 8, y: BY + BH / 2, class: 'sh-tick' }, '1'));

    const sentence = h('div', { class: 'sh-sent', 'aria-live': 'polite' });
    const eq = h('div', { class: 'eq-line' });
    const note = h('p', { class: 'widget-note' });

    function draw(n) {
      const st = steps[n];
      ctx.textContent = `P( w | ${st.prev} )`;
      segs.replaceChildren(...st.d.flatMap((x) => {
        const x1 = X0 + x.lo * BW, w = (x.hi - x.lo) * BW, on = x === st.pick;
        return [
          s('rect', { x: x1, y: BY, width: w, height: BH, class: `sh-seg${on ? ' on' : ''}` }),
          s('text', { x: x1 + w / 2, y: BY + BH / 2, class: `sh-w${on ? ' on' : ''}` }, w > 40 ? x.w : ''),
        ];
      }));
      needle.style.transform = `translateX(${X0 + st.r * BW}px)`;
      needle.querySelector('text').textContent = `r = ${st.r.toFixed(2)}`;

      sentence.replaceChildren(h('span', { class: 'sh-tok bos' }, BOS),
        ...steps.slice(0, n + 1).map((x, i) => h('span', { class: `sh-tok${i === n ? ' cur' : ''}${x.pick.w === EOS ? ' bos' : ''}` }, x.pick.w)));
      const used = steps.slice(0, n + 1);
      const p = used.reduce((a, x) => a * x.pick.p, 1);
      const tex = `P = ${used.map((x) => frac(x.pick)).join('\\cdot ')} = ${p < 0.001 ? p.toExponential(2) : p.toFixed(3)}`;
      katex.render(tex, eq, { throwOnError: false });
      note.textContent = st.pick.w === EOS
        ? 'È uscito </s>: la frase è finita, e la sua probabilità è il prodotto delle probabilità dei bigrammi scelti.'
        : `Dopo «${st.prev}» il corpus contiene ${st.d.map((x) => `«${x.w}» ${x.c} volt${x.c === 1 ? 'a' : 'e'}`).join(' e ')}: r = ${st.r.toFixed(2)} cade nell'intervallo di «${st.pick.w}».`;
    }

    const player = createPlayer({ steps: steps.length, onStep: draw, interval });
    const again = h('button', {
      type: 'button', class: 'seg-btn',
      onclick: () => { steps = sample(dist, Math.random); player.setSteps(steps.length); player.go(0); player.play(); },
    }, 'Genera un\'altra frase');
    el.append(h('div', { class: 'controls' }, h('div', { class: 'seg' }, again)), svg, sentence, eq, note, player.el);
  },
});
