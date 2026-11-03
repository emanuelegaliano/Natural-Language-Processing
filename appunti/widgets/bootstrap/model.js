// paired bootstrap test sull'accuratezza di due classificatori A e B.
// Ogni documento del test set è uno di quattro esiti: 'ab' (entrambi corretti), 'a' (solo A),
// 'b' (solo B), '-' (entrambi sbagliati)

// generatore pseudo-casuale con seme (mulberry32): stessi campioni a ogni caricamento, e nel PDF
export function rng(seed) {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export const accA = (x) => x.filter((c) => c === 'ab' || c === 'a').length / x.length;
export const accB = (x) => x.filter((c) => c === 'ab' || c === 'b').length / x.length;
// le differenze si arrotondano al centesimo: evita che 0.4 diventi 0.39999 e cambi il conteggio
export const delta = (x) => Math.round((accA(x) - accB(x)) * 100) / 100;

// un test set virtuale: n estrazioni con reinserimento dal test set originale
export const resample = (x, rand) => x.map(() => x[Math.floor(rand() * x.length)]);

// b test set virtuali e il p-value: la frazione di campioni con delta >= 2 delta(x)
export function bootstrap(x, b, seed) {
  const rand = rng(seed), d0 = delta(x);
  const samples = [], deltas = [];
  let s = 0;
  for (let i = 0; i < b; i++) {
    const xi = resample(x, rand);
    const di = delta(xi);
    if (i < 50) samples.push(xi);
    deltas.push(di);
    if (di >= 2 * d0 - 1e-9) s++;
  }
  return { d0, samples, deltas, s, p: s / b };
}
