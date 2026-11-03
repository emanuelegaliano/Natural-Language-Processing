import { defineWidget, h, createPlayer } from 'texweb/client';
import './style.css';

// i tentativi per trovare l'articolo "the" in una frase: a ogni passo un pattern, con i match corretti,
// i falsi positivi e i falsi negativi rispetto alle occorrenze vere (date dal pattern gold);
// nel campo di testo si può provare un pattern qualsiasi sulla stessa frase

const PATTERNS = [
  { re: 'the', note: 'Il pattern più semplice: perde il The a inizio frase e trova "the" dentro other e theology.' },
  { re: '[tT]he', note: 'Con la disgiunzione recuperiamo il The maiuscolo, ma i falsi positivi restano.' },
  { re: '[^a-zA-Z][tT]he[^a-zA-Z]', note: 'Un carattere non alfabetico prima e dopo elimina i falsi positivi, ma il The iniziale non ha nessun carattere prima: lo perdiamo di nuovo.' },
  { re: '\\b[tT]he\\b', note: 'I confini di parola non consumano caratteri: funzionano anche a inizio e fine riga.' },
];

// tutti i match di un pattern, come intervalli [inizio, fine)
function matches(src, text) {
  let re;
  try { re = new RegExp(src, 'g'); } catch { return null; }
  const out = [];
  for (const m of text.matchAll(re)) {
    if (m[0].length === 0) continue; // un match vuoto non evidenzia niente
    out.push([m.index, m.index + m[0].length]);
  }
  return out;
}

export default defineWidget({
  name: 'regex-match',
  params: {
    text: { type: 'string', default: 'The other day the theology class was the best.' },
    gold: { type: 'string', default: '\\b[tT]he\\b' },
    interval: { type: 'number', default: 2600, min: 600 },
  },

  mount(el, { text, gold, interval }) {
    el.classList.add('w-regex');
    const truth = matches(gold, text) ?? [];

    const patternLine = h('div', { class: 'rx-pattern' });
    const textLine = h('p', { class: 'rx-text', 'aria-live': 'polite' });
    const stats = h('div', { class: 'rx-stats' });
    const note = h('p', { class: 'widget-note' });

    // una frase con i match evidenziati: ogni carattere sa se è dentro un match, un'occorrenza vera o entrambi
    function show(src, comment) {
      const found = matches(src, text);
      patternLine.replaceChildren(h('span', { class: 'rx-label' }, 'pattern'), h('code', {}, src || ' '));
      if (!found) {
        textLine.replaceChildren(text);
        stats.textContent = '';
        note.textContent = 'Il pattern non è un\'espressione regolare valida.';
        return;
      }
      // un match è corretto se contiene un'occorrenza vera, un'occorrenza è persa se nessun match la contiene
      const ok = found.map(([a, b]) => truth.some(([c, d]) => a <= c && d <= b));
      const missed = truth.map(([c, d]) => !found.some(([a, b]) => a <= c && d <= b));
      const parts = [];
      let i = 0;
      const events = [...found.map(([a, b], k) => ({ a, b, cls: ok[k] ? 'tp' : 'fp' })),
        ...truth.filter((_, k) => missed[k]).map(([a, b]) => ({ a, b, cls: 'fn' }))].sort((x, y) => x.a - y.a);
      for (const ev of events) {
        if (ev.a < i) continue; // match sovrapposti: si evidenzia solo il primo
        if (ev.a > i) parts.push(text.slice(i, ev.a));
        parts.push(h('mark', { class: ev.cls }, text.slice(ev.a, ev.b)));
        i = ev.b;
      }
      parts.push(text.slice(i));
      textLine.replaceChildren(...parts);

      const tp = ok.filter(Boolean).length, fp = ok.length - tp, fn = missed.filter(Boolean).length;
      const cell = (cls, name, v) => h('span', { class: `rx-stat ${cls}` }, [h('b', {}, String(v)), ` ${name}`]);
      stats.replaceChildren(cell('tp', tp === 1 ? 'match corretto' : 'match corretti', tp),
        cell('fp', fp === 1 ? 'falso positivo' : 'falsi positivi', fp),
        cell('fn', fn === 1 ? 'falso negativo' : 'falsi negativi', fn));
      note.textContent = comment ?? (fp + fn === 0 ? 'Tutte e sole le occorrenze di the: nessun errore.' : '');
    }

    const draw = (n) => {
      input.value = PATTERNS[n].re;
      show(PATTERNS[n].re, PATTERNS[n].note);
    };

    // il campo di testo: scrivere un pattern mette in pausa i passi
    const input = h('input', {
      type: 'text', class: 'rx-input', spellcheck: 'false', 'aria-label': 'Prova un pattern',
      oninput: (e) => { player.pause(); show(e.target.value); },
    });
    const custom = h('label', { class: 'rx-custom' }, [h('span', { class: 'ctl-label' }, 'Prova un pattern:'), input]);

    const player = createPlayer({ steps: PATTERNS.length, onStep: draw, interval });
    el.append(patternLine, textLine, stats, note, custom, player.el);
  },
});
