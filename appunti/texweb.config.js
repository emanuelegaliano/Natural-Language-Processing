// configurazione della versione web (texweb)
export default {
  main: 'NaturalLanguageProcessing.tex',
  title: 'Riassunti e Appunti di Natural Language Processing',
  author: 'Emanuele Galiano',
  // le segnalazioni dal pulsante «Segnala un errore» finiscono nelle issue di questa repo
  repository: 'emanuelegaliano/Natural-Language-Processing',
  // \argmax e \argmin sono definiti con \DeclareMathOperator*: KaTeX li riceve già, qui solo \PP
  katexMacros: { '\\PP': '\\operatorname{PP}' },
};
