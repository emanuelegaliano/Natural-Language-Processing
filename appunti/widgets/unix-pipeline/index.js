import { defineWidget, h, createPlayer } from 'texweb/client';
import './style.css';

// la pipeline UNIX per contare le parole di un testo, un comando alla volta:
// a ogni passo si vede l'output del comando, che diventa l'input del successivo

const DEFAULT = "Shall I compare thee to a summer's day?\nThou art more lovely and more temperate:\nRough winds do shake the darling buds of May,\nAnd summer's lease hath all too short a date:";

// confronto byte per byte, come sort con LC_ALL=C (le maiuscole vengono prima delle minuscole)
const cmp = (a, b) => (a < b ? -1 : a > b ? 1 : 0);

const STEPS = [
  { cmd: 'cat sonnet.txt', note: 'Il testo di partenza.', run: (t) => t.split('\n') },
  { cmd: "sed -r 's/[^A-Za-z]+/\\n/g'", note: 'Ogni sequenza di caratteri non alfabetici diventa un a capo: una parola per riga. Il genitivo summer\'s si spezza in summer e s.', run: (t) => t.replace(/[^A-Za-z]+/g, '\n').split('\n') },
  { cmd: "sed '/^$/d'", note: 'Si eliminano le righe vuote (qui quella lasciata dai due punti finali).', run: (l) => l.filter((x) => x !== '') },
  { cmd: 'sort', note: 'Si ordinano le righe: le parole uguali finiscono una sotto l\'altra, e le maiuscole vengono prima delle minuscole.', run: (l) => [...l].sort(cmp) },
  { cmd: 'uniq -c', note: 'Si fondono le righe uguali adiacenti, contandole: ogni riga è ora un type con la sua frequenza.', run: (l) => l.reduce((acc, x) => {
    const last = acc[acc.length - 1];
    if (last && last.w === x) last.c++; else acc.push({ c: 1, w: x });
    return acc;
  }, []) },
  { cmd: 'sort -n -r', note: 'Si ordina per frequenza decrescente (a parità di frequenza, sort confronta le righe intere, sempre al contrario).', run: (l) => [...l].sort((a, b) => b.c - a.c || cmp(b.w, a.w)) },
];

export default defineWidget({
  name: 'unix-pipeline',
  params: {
    text: { type: 'string', default: DEFAULT },
    interval: { type: 'number', default: 2600, min: 600 },
  },

  mount(el, { text, interval }) {
    el.classList.add('w-unix');
    // gli output di tutti i passi, calcolati una volta: il passo 0 parte dal testo, gli altri dal passo precedente
    const outputs = [];
    STEPS.forEach((s, i) => outputs.push(s.run(i <= 1 ? text : outputs[i - 1])));

    const cmds = STEPS.map((s, i) => h('div', { class: 'ux-cmd' }, [h('span', { class: 'ux-pipe' }, i ? '|' : '$'), h('code', {}, s.cmd)]));
    const out = h('div', { class: 'ux-out' });
    const count = h('div', { class: 'ux-count' });
    const note = h('p', { class: 'widget-note' });

    function draw(n) {
      cmds.forEach((c, i) => { c.classList.toggle('on', i === n); c.classList.toggle('done', i < n); });
      const lines = outputs[n];
      out.classList.toggle('text', n === 0);
      out.replaceChildren(...lines.map((x) => h('div', { class: 'ux-line' },
        typeof x === 'string' ? (x === '' ? ' ' : x) : [h('span', { class: 'ux-c' }, String(x.c)), ` ${x.w}`])));
      const words = outputs[2].length, types = outputs[4].length;
      count.textContent = n === 0 ? '' : n < 4 ? `${lines.length} righe` : `${types} types su ${words} token`;
      note.textContent = STEPS[n].note;
    }

    const player = createPlayer({ steps: STEPS.length, onStep: draw, interval });
    el.append(h('div', { class: 'ux-grid' }, [h('div', { class: 'ux-cmds' }, cmds), h('div', {}, [out, count])]), note, player.el);
  },
});
