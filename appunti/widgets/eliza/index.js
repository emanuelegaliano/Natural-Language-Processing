import { defineWidget, h } from 'texweb/client';
import './style.css';

// ELIZA come cascata di sostituzioni: maiuscolo e scambio dei pronomi, poi la prima regola che fa match.
// Le regole usano \b al posto degli spazi delle slide, così funzionano anche a inizio e fine frase

const SWAPS = [
  [/\bI'M\b/g, 'YOU ARE'],
  [/\bMY\b/g, 'YOUR'],
  [/\bME\b/g, 'YOU'],
];

const RULES = [
  { re: /.*\bYOU ARE (DEPRESSED|SAD)\b.*/, out: 'I AM SORRY TO HEAR YOU ARE $1', show: '.* YOU ARE (DEPRESSED|SAD) .*' },
  { re: /.*\bI AM (DEPRESSED|SAD)\b.*/, out: 'WHY DO YOU THINK YOU ARE $1', show: '.* I AM (DEPRESSED|SAD) .*' },
  { re: /.*\bALL\b.*/, out: 'IN WHAT WAY?', show: '.* ALL .*' },
  { re: /.*\bALWAYS\b.*/, out: 'CAN YOU THINK OF A SPECIFIC EXAMPLE?', show: '.* ALWAYS .*' },
  { re: /.*\bI NEED ([^,.!?]*).*/, out: 'WHAT WOULD IT MEAN TO YOU IF YOU GOT $1?', show: '.* I NEED (.*)' },
  { re: /.*\b(YOUR [^,.!?]*).*/, out: '$1', show: '.* (YOUR .*)' },
  { re: /.*/, out: 'PLEASE GO ON', show: 'altrimenti' },
];

const EXAMPLES = [
  'Men are all alike.',
  "They're always bugging us about something or other.",
  'Well, my boyfriend made me come here.',
  "He says I'm depressed much of the time.",
  'I need some help.',
];

// la risposta di ELIZA e i passaggi intermedi
function reply(line) {
  const upper = line.replace(/[’‘]/g, "'").toUpperCase().trim();
  const swapped = SWAPS.reduce((acc, [re, to]) => acc.replace(re, to), upper);
  const k = RULES.findIndex((r) => r.re.test(swapped));
  return { upper, swapped, k, out: swapped.replace(RULES[k].re, RULES[k].out) };
}

export default defineWidget({
  name: 'eliza',
  params: {},

  mount(el) {
    el.classList.add('w-eliza');
    const log = h('div', { class: 'el-log', 'aria-live': 'polite' });
    const steps = h('div', { class: 'el-steps' });
    const rules = RULES.map((r, i) => h('li', { class: 'el-rule' }, [
      h('code', {}, r.show), ' → ', h('code', {}, r.out.replace(/\$(\d)/g, '\\$1')),
    ]));
    const list = h('ol', { class: 'el-rules', start: 2 }, rules);

    function say(line) {
      if (!line.trim()) return;
      const r = reply(line);
      log.append(h('p', { class: 'el-user' }, [h('span', {}, 'User'), line]),
        h('p', { class: 'el-bot' }, [h('span', {}, 'ELIZA'), r.out]));
      log.scrollTop = log.scrollHeight;
      rules.forEach((li, i) => li.classList.toggle('on', i === r.k));
      steps.replaceChildren(
        h('div', {}, [h('span', { class: 'el-tag' }, 'maiuscolo'), h('code', {}, r.upper)]),
        h('div', {}, [h('span', { class: 'el-tag' }, 'pronomi'), h('code', {}, r.swapped)]),
        h('div', {}, [h('span', { class: 'el-tag' }, 'regola'), `${r.k + 2}. `, h('code', {}, RULES[r.k].show)]),
      );
    }

    const input = h('input', { type: 'text', class: 'el-input', placeholder: 'Scrivi qualcosa a ELIZA (in inglese)…', 'aria-label': 'Messaggio per ELIZA' });
    const form = h('form', {
      class: 'el-form',
      onsubmit: (e) => { e.preventDefault(); say(input.value); input.value = ''; },
    }, [input, h('button', { type: 'submit', class: 'seg-btn' }, 'Invia')]);
    const examples = h('div', { class: 'el-examples' }, [h('span', { class: 'ctl-label' }, 'Oppure scegli una frase:'),
      ...EXAMPLES.map((ex) => h('button', { type: 'button', class: 'el-ex', onclick: () => say(ex) }, ex))]);

    el.append(
      h('div', { class: 'el-chat' }, [log, form, examples]),
      h('div', { class: 'el-side' }, [h('p', { class: 'el-head' }, '1. maiuscolo e scambio dei pronomi, poi la prima regola che fa match:'), list, steps]),
    );
    say(EXAMPLES[0]);
  },
});
