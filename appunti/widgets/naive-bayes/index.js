import katex from 'katex';
import { defineWidget, h, createPlayer } from 'texweb/client';
import './style.css';

// l'esempio svolto di Naive Bayes multinomiale con add-1 smoothing, un passo alla volta:
// prior, parole sconosciute, likelihood e punteggio delle due classi; la frase di test si può cambiare

const TRAIN = [
  ['-', 'just plain boring'],
  ['-', 'entirely predictable and lacks energy'],
  ['-', 'no surprises and very few laughs'],
  ['+', 'very powerful'],
  ['+', 'the most fun film of the summer'],
];

const tokens = (t) => t.toLowerCase().split(/[^a-z']+/).filter(Boolean);

function train(docs) {
  const classes = ['-', '+'];
  const vocab = new Set(docs.flatMap(([, d]) => tokens(d)));
  const model = {};
  for (const c of classes) {
    const words = docs.filter(([k]) => k === c).flatMap(([, d]) => tokens(d));
    const count = new Map();
    words.forEach((w) => count.set(w, (count.get(w) ?? 0) + 1));
    model[c] = { nDocs: docs.filter(([k]) => k === c).length, nWords: words.length, count };
  }
  return { classes, vocab, model, nDocs: docs.length };
}

// numeri per le formule: interi così come sono, il resto in notazione scientifica se piccolo
const sci = (v) => {
  if (v >= 0.001) return v.toFixed(3);
  const [m, e] = v.toExponential(1).split('e');
  return `${m} \\times 10^{${Number(e)}}`;
};

export default defineWidget({
  name: 'naive-bayes',
  params: {
    test: { type: 'string', default: 'predictable with no fun' },
    interval: { type: 'number', default: 2600, min: 600 },
  },

  mount(el, { test, interval }) {
    el.classList.add('w-nb');
    const nb = train(TRAIN);
    const V = nb.vocab.size;
    const label = { '-': '−', '+': '+' };

    const trainBox = h('table', { class: 'nb-train' }, [
      h('tr', {}, [h('th', {}, 'classe'), h('th', {}, 'documento di training')]),
      ...TRAIN.map(([c, d]) => h('tr', {}, [h('td', { class: `nb-c ${c === '+' ? 'pos' : 'neg'}` }, label[c]), h('td', {}, d)])),
    ]);
    const testLine = h('div', { class: 'nb-test' });
    const table = h('table', { class: 'nb-lik' });
    const eqs = h('div', { class: 'nb-eqs' });
    const note = h('p', { class: 'widget-note' });

    let words = [];
    function draw(n) {
      const known = words.filter((w) => nb.vocab.has(w));
      testLine.replaceChildren(h('span', { class: 'ctl-label' }, 'test: '), ...words.map((w) =>
        h('span', { class: `nb-w${n >= 2 && !nb.vocab.has(w) ? ' drop' : ''}` }, w)));

      // tabella dei conteggi e delle likelihood delle parole di test
      const rowsT = [...new Set(known)].map((w) => h('tr', {}, [
        h('td', { class: 'nb-word' }, w),
        ...nb.classes.flatMap((c) => {
          const m = nb.model[c], cnt = m.count.get(w) ?? 0;
          return [h('td', {}, n >= 3 ? String(cnt) : ''), h('td', { class: 'nb-p' }, n >= 3 ? `(${cnt}+1)/(${m.nWords}+${V}) = ${((cnt + 1) / (m.nWords + V)).toFixed(3)}` : '')];
        }),
      ]));
      table.replaceChildren(h('tr', {}, [h('th', {}, 'parola'), h('th', {}, 'count(w, −)'), h('th', {}, 'P(w | −)'), h('th', {}, 'count(w, +)'), h('th', {}, 'P(w | +)')]), ...rowsT);
      table.classList.toggle('on', n >= 3);

      const lines = [];
      if (n >= 0) lines.push(`|V| = ${V}, \\qquad \\textstyle\\sum_w \\text{count}(w, -) = ${nb.model['-'].nWords}, \\qquad \\sum_w \\text{count}(w, +) = ${nb.model['+'].nWords}`);
      if (n >= 1) lines.push(`P(-) = \\tfrac{${nb.model['-'].nDocs}}{${nb.nDocs}}, \\qquad P(+) = \\tfrac{${nb.model['+'].nDocs}}{${nb.nDocs}}`);
      let best = null;
      if (n >= 4) {
        const score = {};
        for (const c of nb.classes) {
          const m = nb.model[c];
          const nums = known.map((w) => (m.count.get(w) ?? 0) + 1);
          const p = (m.nDocs / nb.nDocs) * nums.reduce((a, k) => a * k / (m.nWords + V), 1);
          score[c] = p;
          const sgn = c === '+' ? '+' : '-';
          lines.push(`P(${sgn})\\,P(S \\mid ${sgn}) = \\tfrac{${m.nDocs}}{${nb.nDocs}} \\times \\frac{${nums.join(' \\times ') || 1}}{${m.nWords + V}^{${known.length}}} = ${sci(p)}`);
        }
        best = score['-'] >= score['+'] ? '-' : '+';
        lines.push(`\\hat c = \\arg\\max_c P(c) \\textstyle\\prod_i P(w_i \\mid c) = ${best === '+' ? '+\\ (\\text{positiva})' : '-\\ (\\text{negativa})'}`);
      }
      eqs.replaceChildren(...lines.map((tex, i) => {
        const d = h('div', { class: `eq-line${i === lines.length - 1 ? ' cur' : ''}` });
        katex.render(tex, d, { throwOnError: false });
        return d;
      }));
      const dropped = words.filter((w) => !nb.vocab.has(w));
      note.textContent = [
        'Il training set ha 5 documenti: il vocabolario è l\'unione delle parole di entrambe le classi.',
        'I prior sono le frazioni di documenti di ciascuna classe.',
        dropped.length ? `${dropped.map((w) => `«${w}»`).join(', ')} non compare nel training: la togliamo dal documento di test.` : 'Tutte le parole di test sono nel vocabolario.',
        'Likelihood con add-1 smoothing: anche le parole mai viste in una classe hanno probabilità positiva.',
        best ? `Il punteggio di ogni classe è il prior per il prodotto delle likelihood: vince la classe ${best === '+' ? 'positiva' : 'negativa'}.` : '',
      ][n] ?? '';
    }

    const player = createPlayer({ steps: 5, onStep: draw, interval });
    const input = h('input', {
      type: 'text', class: 'nb-input', value: test, spellcheck: 'false', 'aria-label': 'Frase di test',
      oninput: (e) => { words = tokens(e.target.value); draw(player.index); },
    });
    words = tokens(test);
    el.append(h('div', { class: 'nb-top' }, [trainBox,
      h('label', { class: 'nb-custom' }, [h('span', { class: 'ctl-label' }, 'Frase di test:'), input])]),
    testLine, table, eqs, note, player.el);
  },
});
