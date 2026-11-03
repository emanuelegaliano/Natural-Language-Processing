import { defineWidget, h, createPlayer } from 'texweb/client';
import { learn, segment, EOW } from './bpe.js';
import './style.css';

// il token learner di BPE un merge alla volta: corpus, vocabolario e coppie adiacenti più frequenti;
// sotto, il token segmenter applica a una parola nuova i merge appresi fino al passo corrente

export default defineWidget({
  name: 'bpe',
  params: {
    words: { type: 'list', default: ['low', 'lowest', 'newer', 'wider', 'new'] },
    counts: { type: 'list', default: [5, 2, 6, 3, 2] },
    merges: { type: 'integer', default: 8, min: 1, max: 30 },
    test: { type: 'string', default: 'lower' },
    interval: { type: 'number', default: 2200, min: 400 },
  },

  mount(el, { words, counts, merges, test, interval }) {
    el.classList.add('w-bpe');
    const corpus = words.map((w, i) => ({ word: String(w), count: Number(counts[i] ?? 1) }));
    const states = learn(corpus, merges);
    const learned = states.slice(1).map((s) => s.merge);

    const chip = (t, cls = '') => h('span', { class: `bp-chip ${cls}` }, t === EOW ? '_' : t);
    const head = h('div', { class: 'eq-line bp-head', 'aria-live': 'polite' });
    const corpusBox = h('div', { class: 'bp-corpus' });
    const vocabBox = h('div', { class: 'bp-vocab' });
    const pairsBox = h('ol', { class: 'bp-pairs' });
    const segOut = h('div', { class: 'bp-seg-out' });

    let index = 0;
    function draw(n) {
      index = n;
      const st = states[n], tok = st.merge ? st.merge.left + st.merge.right : null;
      head.replaceChildren(...(st.merge
        ? [`merge ${n}: `, chip(st.merge.left), ' + ', chip(st.merge.right), ' → ', chip(tok, 'new'), ` (${st.merge.count} occorrenze)`]
        : ['il corpus di partenza: ogni parola è una sequenza di caratteri più il simbolo di fine parola _']));

      corpusBox.replaceChildren(h('div', { class: 'bp-title' }, 'corpus'), ...st.words.map((w) => h('div', { class: 'bp-row' }, [
        h('span', { class: 'bp-count' }, String(w.count)), ...w.syms.map((s) => chip(s, s === tok ? 'new' : '')),
      ])));
      vocabBox.replaceChildren(h('div', { class: 'bp-title' }, 'vocabolario'),
        h('div', { class: 'bp-chips' }, st.vocab.map((t, i) => chip(t, t === tok ? 'new' : i >= states[0].vocab.length ? 'merged' : ''))));

      // le coppie del passo successivo: la prima per frequenza (a parità, la prima incontrata) è il prossimo merge
      const next = states[n + 1]?.merge;
      const top = [...st.pairs].sort((a, b) => b.count - a.count).slice(0, 6);
      pairsBox.replaceChildren(...top.map((p, i) => h('li', { class: next && p.left === next.left && p.right === next.right ? 'best' : '' }, [
        h('span', { class: 'bp-rank' }, `${i + 1}.`), chip(p.left), chip(p.right), h('span', { class: 'bp-pc' }, String(p.count)),
      ])));
      drawSegment();
    }

    // il segmenter con i merge appresi fino al passo corrente
    function drawSegment() {
      const word = input.value.trim().replace(/\s+/g, '');
      if (!word) { segOut.replaceChildren(); return; }
      const { syms } = segment(word, learned.slice(0, index));
      segOut.replaceChildren(...syms.map((s) => chip(s, s.length > 1 ? 'merged' : '')),
        h('span', { class: 'bp-n' }, `${syms.length} token con i primi ${index} merge`));
    }
    const input = h('input', { type: 'text', class: 'bp-input', value: test, spellcheck: 'false', 'aria-label': 'Parola da segmentare', oninput: drawSegment });

    const player = createPlayer({ steps: states.length, onStep: draw, interval });
    el.append(
      head,
      h('div', { class: 'bp-grid' }, [corpusBox, vocabBox,
        h('div', { class: 'bp-pairs-box' }, [h('div', { class: 'bp-title' }, 'coppie adiacenti più frequenti'), pairsBox])]),
      h('label', { class: 'bp-seg' }, [h('span', { class: 'ctl-label' }, 'Token segmenter:'), input, segOut]),
      player.el,
    );
  },
});
