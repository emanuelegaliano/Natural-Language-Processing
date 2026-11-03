// BPE: apprendimento dei merge su un corpus di parole con le loro frequenze, e segmentazione
// di una parola nuova applicando i merge nell'ordine in cui sono stati appresi.
// A parità di frequenza vince la coppia incontrata per prima scorrendo il corpus, come nelle slide

export const EOW = '_';

// corpus: [{ word, count }] -> lista di stati, uno per merge (lo stato 0 è il corpus di caratteri)
export function learn(corpus, k) {
  let words = corpus.map(({ word, count }) => ({ syms: [...word, EOW], count }));
  const vocab = [...new Set(words.flatMap((w) => w.syms))].sort((a, b) => (a === EOW ? -1 : b === EOW ? 1 : a.localeCompare(b)));
  const states = [{ words, vocab: [...vocab], merge: null, pairs: countPairs(words) }];
  for (let i = 0; i < k; i++) {
    const pairs = countPairs(words);
    if (!pairs.length) break;
    const best = pairs.reduce((m, p) => (p.count > m.count ? p : m));
    const tok = best.left + best.right;
    words = words.map((w) => ({ ...w, syms: apply(w.syms, best.left, best.right) }));
    vocab.push(tok);
    states.push({ words, vocab: [...vocab], merge: best, pairs: countPairs(words) });
  }
  return states;
}

// coppie adiacenti con la frequenza totale, nell'ordine in cui compaiono nel corpus
export function countPairs(words) {
  const map = new Map();
  for (const w of words) {
    for (let j = 0; j + 1 < w.syms.length; j++) {
      const key = `${w.syms[j]}\u0000${w.syms[j + 1]}`;
      if (!map.has(key)) map.set(key, { left: w.syms[j], right: w.syms[j + 1], count: 0 });
      map.get(key).count += w.count;
    }
  }
  return [...map.values()];
}

// sostituisce ogni coppia adiacenti (left, right) con il simbolo unito
export function apply(syms, left, right) {
  const out = [];
  for (let j = 0; j < syms.length; j++) {
    if (j + 1 < syms.length && syms[j] === left && syms[j + 1] === right) { out.push(left + right); j++; }
    else out.push(syms[j]);
  }
  return out;
}

// segmentazione: i caratteri della parola più il simbolo di fine parola, poi i merge in ordine
export function segment(word, merges) {
  let syms = [...word, EOW];
  const trace = [syms];
  for (const m of merges) {
    const next = apply(syms, m.left, m.right);
    if (next.length !== syms.length) trace.push(next);
    syms = next;
  }
  return { syms, trace };
}
