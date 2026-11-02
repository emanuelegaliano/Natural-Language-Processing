# The Linear Representation Hypothesis and the Geometry of Large Language Models

Questi sono gli appunti di studio sul paper di Park, Choe e Veitch (ICML 2024). L'idea di partenza è la **linear representation hypothesis**, ovvero l'ipotesi che i concetti di alto livello (il genere, la lingua, il tempo verbale) siano rappresentati come direzioni nello spazio delle rappresentazioni di un LLM. Il paper si pone due domande: cosa vuol dire, formalmente, "rappresentazione lineare", e con quale prodotto interno conviene misurare le direzioni.

Gli appunti seguono il paper sezione per sezione e vengono estesi man mano che lo studio va avanti. I chiarimenti emersi durante lo studio sono già integrati nel testo. Il PDF del paper è in `paper/refs/The linear hypothesis.pdf`.

## Indice

- [Notazione di partenza](#notazione-di-partenza)
- [Sezione 2: The Linear Representation Hypothesis](#sezione-2-the-linear-representation-hypothesis)
  - [2.1 Concetti](#21-concetti)
  - [2.2 Unembedding representation e measurement](#22-unembedding-representation-e-measurement)
  - [2.3 Embedding representation e intervention](#23-embedding-representation-e-intervention)
- [Sezione 3: Inner Product for Language Model Representations](#sezione-3-inner-product-for-language-model-representations)
  - [Ripasso di algebra lineare](#ripasso-di-algebra-lineare)
  - [Preliminari](#preliminari)
  - [Non identificabilità del prodotto interno](#non-identificabilità-del-prodotto-interno)
  - [Prodotto interno causale (Definizione 3.1)](#prodotto-interno-causale-definizione-31)
  - [Teorema 3.2 (unificazione delle rappresentazioni)](#teorema-32-unificazione-delle-rappresentazioni)
  - [Assunzione 3.3](#assunzione-33)
  - [Teorema 3.4 (forma esplicita)](#teorema-34-forma-esplicita)
  - [Gradi di libertà e scelta di D](#gradi-di-libertà-e-scelta-di-d)
  - [Rappresentazioni unificate](#rappresentazioni-unificate)
  - [Verifica finale](#verifica-finale)

---

## Notazione di partenza

Un language model prende un contesto $x$ e genera l'output una parola (token) alla volta. Il **contesto** è tutto il testo che il modello ha già visto prima della parola da predire, ovvero il prompt più i token già generati. Ad esempio, se il modello deve completare "She is the monarch of", il contesto $x$ è proprio "She is the monarch of", e la parola da predire sarà "England". Ogni volta che il modello genera una parola, il contesto si allunga di un token.

Per generare la parola successiva, il modello lavora con due tipi di vettori, che vivono in due spazi diversi della stessa dimensione $d$:

- **Embedding $\lambda(x) \in \Lambda \simeq \mathbb{R}^d$**: rappresenta il contesto in input. Tutta la sequenza passa attraverso i layer del transformer, e alla fine resta un solo vettore: l'hidden state dell'ultimo layer nella posizione dell'ultimo token. Quindi $\lambda(x)$ dipende da tutto il prefisso, non solo dall'ultima parola, e va ricalcolato a ogni nuovo token. Ad esempio, "He is the monarch of" e "She is the monarch of" finiscono su due punti diversi di $\Lambda$, pur differendo per un solo token.
- **Unembedding $\gamma(y) \in \Gamma \simeq \mathbb{R}^d$**: rappresenta una parola $y$ del vocabolario in output. Ce n'è uno solo per ogni parola, ed è fisso: è la riga di $y$ nella matrice di output (il "LM head", quella che sta prima della softmax). Con un vocabolario di 32K token avremo esattamente 32K vettori $\gamma$, che dopo il training non cambiano più.

Attenzione al nome: nel paper "embedding" non è la matrice di input, come si intende di solito, ma l'ultimo hidden state del modello. Conviene quindi ricordare una sola cosa: $\lambda$ = contesti, $\gamma$ = parole. Il nome unembedding viene dal fatto che fa il percorso inverso dell'embedding di input: quest'ultimo porta un token dentro lo spazio vettoriale, l'unembedding riporta un vettore a un token.

Il modello non sceglie subito una parola. Prima calcola una distribuzione $P(y \mid x)$ su tutto il vocabolario, ovvero la probabilità che la parola successiva sia proprio $y$, dato il contesto $x$. Ad esempio, con $x$ = "She is the monarch of England, the", "queen" avrà probabilità alta, "king" più bassa e "banana" quasi zero. Poi campiona da questa distribuzione, oppure prende la parola più probabile. La distribuzione è data dalla softmax:

$$P(y \mid x) \propto \exp\big(\lambda(x)^\top \gamma(y)\big)$$

dove il simbolo $\propto$ indica che manca la costante di normalizzazione. Scritta per intero, la formula è:

$$P(y \mid x) = \frac{\exp\big(\lambda(x)^\top \gamma(y)\big)}{\sum_{y'} \exp\big(\lambda(x)^\top \gamma(y')\big)}$$

dove il numeratore è il punteggio della parola $y$ e il denominatore somma i punteggi di tutte le parole $y'$ del vocabolario, così le probabilità sommano a 1.

Il prodotto scalare $\lambda(x)^\top \gamma(y)$ non è un'assunzione degli autori, ma viene direttamente dall'architettura. L'ultimo layer di un language model decoder-only (GPT, LLaMA) è un layer lineare che porta l'hidden state ai logit, tramite la matrice di output $U \in \mathbb{R}^{|V| \times d}$:

$$\text{logits} = U\,\lambda(x)$$

dove $|V|$ è la dimensione del vocabolario e il risultato è un vettore con un punteggio per ogni parola. La componente relativa a $y$ è il prodotto tra la riga $y$ di $U$ e $\lambda(x)$. Quella riga è appunto $\gamma(y)$, quindi il logit di $y$ vale $\lambda(x)^\top \gamma(y)$. Applicando la softmax a questo vettore si ottiene la formula del paper, che quindi si limita a dare un nome alle due parti del prodotto.

L'intuizione è quella della similarità: il prodotto scalare è grande quando i due vettori sono allineati. In pratica $\lambda(x)$ dice "cosa voglio dire adesso" e ogni $\gamma(y)$ dice "cosa significa questa parola come output". Una parola è quindi probabile quando il suo $\gamma(y)$ punta nella stessa direzione di $\lambda(x)$. Proprio perché il collegamento tra contesto e parola è un prodotto scalare, ha senso parlare di direzioni e di rappresentazioni lineari: spostare $\lambda$ lungo una direzione cambia i logit in modo lineare, ed è quello che sfrutteranno il Teorema 2.2 e lo steering.

La distinzione tra i due spazi è il motivo per cui la sezione 2 ha due definizioni. La Definizione 2.1 guarda le differenze tra parole, come $\gamma(\text{queen}) - \gamma(\text{king})$. La 2.3 invece guarda le differenze tra contesti, come $\lambda(\text{"She is..."}) - \lambda(\text{"He is..."})$. Inoltre, il Teorema 2.2 ragionerà su $P(y \mid x)$ restretta a due sole parole: non "quanto è probabile queen tra tutte le parole", ma "sapendo che l'output è king oppure queen, quanto è probabile queen".

## Sezione 2: The Linear Representation Hypothesis

### 2.1 Concetti

Il primo passo è definire che cos'è un concetto. Intuitivamente, un concetto è un fattore di variazione che può essere cambiato in isolamento: possiamo passare da un output in inglese a uno in francese senza cambiarne il significato, oppure da un testo che parla di un uomo a uno che parla di una donna senza cambiare lingua.

Il paper formalizza questa idea con un modello causale. Un **concetto** $W$ è una variabile latente causata dal contesto $X$, che a sua volta causa l'output $Y$:

$$X \to W \to Y$$

Ad esempio, il contesto "She is the monarch of" rende probabile il valore femminile del concetto di genere, e questo a sua volta rende probabile l'output "queen".

Per semplicità ci si limita a concetti **binari**, a cui viene dato un verso: scriviamo male⇒female, con $W=0$ per maschile e $W=1$ per femminile. Il verso serve perché più avanti i concetti diventeranno vettori, e vogliamo che il segno abbia un significato: la rappresentazione di female⇒male sarà l'opposta di quella di male⇒female.

Ogni concetto $W$ definisce un insieme di **output controfattuali** $\{Y(W=w)\}$, ovvero output che differiscono soltanto nel valore di $W$. Per male⇒female, la coppia $(Y(0), Y(1))$ è un elemento casuale dell'insieme:

$$\{(\text{man}, \text{woman}),\ (\text{king}, \text{queen}),\ \dots\}$$

Il termine controfattuale viene dall'inferenza causale: $Y(1)$ è la parola che avremmo ottenuto se, a parità di tutto il resto, il concetto avesse avuto valore 1.

Il paper fa poi un'assunzione: il valore del concetto si legge in modo deterministico dall'output. Se l'output è "king", allora $W=0$. Ne consegue che per specificare un concetto basta specificare le sue coppie controfattuali. In pratica, un concetto è una lista di coppie di parole, ed è proprio così che vengono costruiti i concetti negli esperimenti (con il dataset BATS).

Più avanti servirà ragionare su più concetti insieme. Due concetti $W$ e $Z$ si dicono **causalmente separabili** se l'output $Y(W=w, Z=z)$ è ben definito per ogni coppia di valori $w, z$. In altre parole, i due concetti possono variare liberamente e indipendentemente l'uno dall'altro. Quando i concetti sono chiari dal contesto, scriveremo semplicemente $Y(w,z)$.

Torniamo al nostro esempio. Con $W$ = male⇒female e $Z$ = English⇒French, tutte e quattro le combinazioni esistono:

| | $Z=0$ (inglese) | $Z=1$ (francese) |
|---|---|---|
| $W=0$ (maschile) | king | roi |
| $W=1$ (femminile) | queen | reine |

quindi i due concetti sono causalmente separabili. Genere e lingua sono due proprietà diverse della parola: possiamo scegliere un valore per ciascuna in modo indipendente, e ogni cella contiene una parola che esiste.

Al contrario, English⇒French e English⇒Russian descrivono la stessa proprietà, ovvero la lingua. Prendiamo $W$ = English⇒French e $Z$ = English⇒Russian e proviamo a riempire la tabella:

| | $Z=0$ (inglese) | $Z=1$ (russo) |
|---|---|---|
| $W=0$ (inglese) | king | ? (inglese e russa) |
| $W=1$ (francese) | ? (francese e inglese) | ? (francese e russa) |

L'unica cella sensata è $Y(0,0)$: le altre tre richiederebbero una parola in due lingue contemporaneamente. In pratica i due concetti non possono variare liberamente, perché spostando $W$ verso il francese si sposta per forza anche $Z$ (la parola non è più inglese). Quindi non sono causalmente separabili.

Un'osservazione che tornerà utile: la separabilità causale non ha niente a che fare con la correlazione nei dati. Anche se nei testi in francese si parlasse più spesso di uomini, genere e lingua resterebbero separabili, perché "reine" esiste comunque.

### 2.2 Unembedding representation e measurement

Abbiamo due spazi di rappresentazione, $\Lambda$ (contesti) e $\Gamma$ (parole), e un concetto potrebbe essere rappresentato linearmente in ciascuno dei due. Il paper parte da $\Gamma$.

Si potrebbe pensare di formalizzare l'idea classica dei word embedding, ovvero che $\gamma(\text{queen}) - \gamma(\text{king})$ e $\gamma(\text{woman}) - \gamma(\text{man})$ appartengano allo stesso sottospazio di dimensione 1. L'intuizione non sarebbe nemmeno sbagliata, ma un sottospazio contiene sia $v$ che $-v$. Quindi anche $\gamma(\text{king}) - \gamma(\text{queen})$ starebbe nello stesso sottospazio, pur andando nel verso opposto: il verso del concetto andrebbe perso.

Possiamo risolvere il problema sostituendo il sottospazio con un cono. Il nome inganna: qui "cono" non indica la forma geometrica, ma in matematica si chiama cono qualsiasi insieme che resta chiuso rispetto alla moltiplicazione per numeri positivi. Il **cono** di un vettore $v$ è l'insieme dei suoi multipli positivi:

$$\text{Cone}(v) = \{\alpha v : \alpha > 0\}$$

dove la condizione $\alpha > 0$ fa sì che non si ottenga una retta, ma una **semiretta** che parte dall'origine e passa per $v$. In pratica il cono di un singolo vettore è una direzione con il verso.

L'intuizione è che un concetto sia una freccia: maschile→femminile punta in una direzione precisa dello spazio. Non conta quanto ci si sposta lungo la freccia, conta in che direzione e in che verso. Facciamo un esempio in 2D, supponendo che:

- queen − king = (2, 2)
- woman − man = (1, 1)
- reine − roi = (3, 3)

Sono tre lunghezze diverse sulla stessa semiretta, quella di $v = (1,1)$, quindi tutte e tre stanno in $\text{Cone}(v)$, con $\alpha = 2, 1, 3$. Invece king − queen = (−2, −2) sta sulla stessa retta, ma sulla metà opposta. Un sottospazio lo accetterebbe, la semiretta no. Ed è giusto rifiutarlo, perché va da femminile a maschile, ovvero rappresenta il concetto opposto.

Per capire perché serve proprio il cono, possiamo confrontare tre insiemi in cui potremmo chiedere che cadano le differenze controfattuali:

- **Un singolo vettore** $\{\bar\gamma_W\}$: è troppo rigido, perché queen−king e reine−roi dovrebbero essere identiche, anche in lunghezza.
- **La retta** $\{\alpha \bar\gamma_W : \alpha \in \mathbb{R}\}$: è troppo larga, perché accetta anche king−queen, che va nel verso opposto.
- **Il cono** $\{\alpha \bar\gamma_W : \alpha > 0\}$: contiene tutti i multipli positivi e nient'altro. Le lunghezze possono variare, il verso no.

Partendo dal singolo vettore e volendo permettere lunghezze diverse, bisogna aggiungere tutti i suoi multipli positivi, e si ottiene proprio il cono. Aggiungendo qualsiasi altro vettore, si accetterebbe un verso diverso oppure si uscirebbe dalla direzione. Quindi il cono è l'insieme minimo che fa esattamente quello che serve.

Il vincolo $\alpha > 0$ è anche quello che fa funzionare il Teorema 2.2, che vedremo tra poco. Se $\alpha$ potesse essere negativo, per alcune coppie un valore alto di $\lambda^\top \bar\gamma_W$ alzerebbe la probabilità di $Y(1)$, e per altre la abbasserebbe. A quel punto non avremmo più un'unica direzione che misura il concetto.

**Definizione 2.1 (unembedding representation).** Un vettore $\bar\gamma_W$ è una rappresentazione di unembedding del concetto $W$ se:

$$\gamma(Y(1)) - \gamma(Y(0)) \in \text{Cone}(\bar\gamma_W) \quad \text{quasi certamente}$$

dove "quasi certamente" si riferisce alla coppia controfattuale $(Y(0), Y(1))$, che è casuale: la condizione deve valere per tutte le coppie del concetto, a meno di un insieme di probabilità nulla.

Nel nostro esempio, $\gamma(\text{queen}) - \gamma(\text{king})$, $\gamma(\text{woman}) - \gamma(\text{man})$ e $\gamma(\text{reine}) - \gamma(\text{roi})$ devono puntare tutte nella stessa direzione e nello stesso verso, anche con lunghezze diverse. Se esiste, $\bar\gamma_W$ è unico a meno di una costante positiva: è appunto una direzione, coerente con l'idea che i concetti siano rappresentati come direzioni.

Il primo risultato della sezione collega questa definizione alla nozione di measurement, ovvero al probing lineare.

**Teorema 2.2 (measurement representation).** Sia $\bar\gamma_W$ la rappresentazione di unembedding di $W$. Allora, per ogni embedding di contesto $\lambda \in \Lambda$:

$$\text{logit}\, P\big(Y = Y(1) \mid Y \in \{Y(0), Y(1)\}, \lambda\big) = \alpha\, \lambda^\top \bar\gamma_W$$

dove $\text{logit}(p) = \log \frac{p}{1-p}$ e $\alpha > 0$ è uno scalare che dipende dalla coppia $\{Y(0), Y(1)\}$.

La dimostrazione (Appendice B.1) è molto banale. Condizionando sull'evento $Y \in \{Y(0), Y(1)\}$, restano in gioco solo due parole. Scriviamo $\gamma_0 = \gamma(Y(0))$ e $\gamma_1 = \gamma(Y(1))$. La probabilità condizionata è:

$$P\big(Y = Y(1) \mid Y \in \{Y(0), Y(1)\}, \lambda\big) = \frac{\exp(\lambda^\top \gamma_1)}{\exp(\lambda^\top \gamma_0) + \exp(\lambda^\top \gamma_1)}$$

dove la costante di normalizzazione della softmax si è già cancellata, in quanto compare sia al numeratore che al denominatore. Il logit è il logaritmo del rapporto tra le probabilità delle due parole:

$$\text{logit}\, P(\cdots) = \log \frac{\exp(\lambda^\top \gamma_1)}{\exp(\lambda^\top \gamma_0)} = \lambda^\top (\gamma_1 - \gamma_0) = \alpha\, \lambda^\top \bar\gamma_W$$

dove l'ultimo passaggio è proprio la Definizione 2.1: $\gamma_1 - \gamma_0$ sta nel cono di $\bar\gamma_W$, quindi è uguale ad $\alpha\bar\gamma_W$ per qualche $\alpha > 0$.

Cosa dice, in pratica, il teorema? Supponiamo che il contesto parli di un monarca, quindi sappiamo che l'output sarà "king" oppure "queen". Allora la probabilità che sia "queen" è:

$$P = \sigma(\alpha\, \lambda^\top \bar\gamma_W)$$

dove $\sigma$ è la sigmoide. È esattamente il modello della regressione logistica, con feature $\lambda$ e vettore dei pesi $\bar\gamma_W$ (a meno di $\alpha$)[^1]. Lo scalare $\alpha$ può cambiare da coppia a coppia (king/queen e roi/reine avranno $\alpha$ diversi), ma la direzione usata per la predizione è la stessa per tutte le coppie del concetto.

Il legame con il probing è il seguente. Un **probe lineare** è un classificatore lineare (tipicamente una regressione logistica) addestrato a prevedere il valore di un concetto dalle rappresentazioni del modello. Anche il probe produce un predittore lineare sulla scala del logit, quindi le due nozioni hanno la stessa forma.

C'è però una differenza. Il probe impara dai dati, quindi può assorbire anche concetti correlati ma diversi da quello bersaglio. Ad esempio, se i testi in francese parlassero in maggioranza di uomini, un probe per English⇒French imparerebbe anche un pezzo della direzione male⇒female. La rappresentazione di unembedding, invece, è costruita da coppie che differiscono solo in $W$, quindi questa informazione non entra. Per questo il paper la considera una rappresentazione di probing **ideale**.

La 2.2 collega quindi la prima nozione dell'introduzione (subspace, nello spazio delle parole) alla seconda (measurement). La 2.3 farà lo stesso nello spazio dei contesti, collegando subspace e intervention.

### 2.3 Embedding representation e intervention

Passiamo ora allo spazio $\Lambda$ dei contesti. Anche qui vogliamo una definizione basata su coppie, ma questa volta le coppie non sono di parole: sono di contesti.

Ogni embedding $\lambda(x)$ induce una distribuzione sui concetti: il contesto "She is the monarch of England," rende probabile che la parola successiva sia al femminile. Consideriamo quindi coppie di contesti come:

$$\lambda_0 = \lambda(\text{"He is the monarch of England, "}), \qquad \lambda_1 = \lambda(\text{"She is the monarch of England, "})$$

dove i due contesti cambiano la distribuzione sul genere (il concetto bersaglio), ma lasciano invariata quella su tutti gli altri concetti, ad esempio la lingua. Un concetto è rappresentato nello spazio di embedding se le differenze tra tutte le coppie di questo tipo stanno nello stesso cono.

**Definizione 2.3 (embedding representation).** Un vettore $\bar\lambda_W$ è una rappresentazione di embedding del concetto $W$ se $\lambda_1 - \lambda_0 \in \text{Cone}(\bar\lambda_W)$ per ogni coppia di embedding $\lambda_0, \lambda_1 \in \Lambda$ che soddisfi:

$$\frac{P(W = 1 \mid \lambda_1)}{P(W = 1 \mid \lambda_0)} > 1 \qquad \text{e} \qquad \frac{P(W, Z \mid \lambda_1)}{P(W, Z \mid \lambda_0)} = \frac{P(W \mid \lambda_1)}{P(W \mid \lambda_0)}$$

per ogni concetto $Z$ causalmente separabile da $W$.

I concetti $Z$ causalmente separabili da $W$ sono tutti gli altri concetti indipendenti dal genere, come la lingua, il tempo verbale o il plurale. Ricordando la 2.1, sono i concetti che possono variare liberamente insieme a $W$, come genere e lingua nella tabella king/roi/queen/reine. La coppia di contesti può spostare $W$, ma non deve spostare nessuno di questi. I concetti non separabili da $W$ invece non vengono controllati, perché non si possono tenere fermi mentre $W$ cambia.

Le due condizioni hanno un ruolo preciso: la prima dice che la coppia è rilevante per il concetto bersaglio, la seconda che non tocca nessun altro concetto.

La **prima condizione** è più semplice di come appare. Un rapporto è maggiore di 1 se e solo se il numeratore è più grande del denominatore, quindi la condizione equivale a:

$$P(W = 1 \mid \lambda_1) > P(W = 1 \mid \lambda_0)$$

ovvero, nel contesto $\lambda_1$ il valore $W=1$ è più probabile che nel contesto $\lambda_0$. Scriverla come rapporto serve solo a metterla nella stessa forma della seconda condizione. Torniamo al nostro esempio, in cui $W=1$ è il femminile:

- $\lambda_0$ = "He is the monarch of England, the": la parola successiva sarà quasi sicuramente maschile (king), quindi $P(W=1 \mid \lambda_0)$ è bassa, ad esempio 0.05.
- $\lambda_1$ = "She is the monarch of England, the": la parola successiva sarà quasi sicuramente femminile (queen), quindi $P(W=1 \mid \lambda_1)$ è alta, ad esempio 0.95.

Il rapporto vale $0.95 / 0.05 = 19 > 1$, quindi la coppia soddisfa la condizione. Che $W=1$ diventi più probabile non è qualcosa che si dimostra, ma un requisito: tra tutte le coppie di contesti, la definizione considera solo quelle in cui passare da $\lambda_0$ a $\lambda_1$ sposta il concetto verso $W=1$.

Il requisito serve innanzitutto a fissare il verso. Se prendessimo la coppia al contrario ("She" come $\lambda_0$ e "He" come $\lambda_1$), il rapporto varrebbe $0.05/0.95 < 1$ e la coppia verrebbe scartata. Senza questa condizione, nel cono di $\bar\lambda_W$ finirebbero sia "She − He" che "He − She", e torneremmo al problema del sottospazio visto nella 2.2. Inoltre, la condizione esclude le coppie che non c'entrano niente con il genere. Ad esempio, "He is the monarch of England" e "He is the monarch of France" hanno rapporto circa 1, quindi non dicono niente su $\bar\lambda_W$.

La **seconda condizione** è la meno immediata. Dice che tutto quello che cambia nella distribuzione congiunta di $W$ e $Z$ è spiegato dal cambiamento di $W$. Applicando la regola della catena $P(W, Z \mid \lambda) = P(W \mid \lambda)\,P(Z \mid W, \lambda)$, il rapporto si semplifica e la condizione diventa:

$$P(Z \mid W, \lambda_1) = P(Z \mid W, \lambda_0)$$

ovvero, a genere fissato, la distribuzione sulla lingua è la stessa nei due contesti. Torniamo al nostro esempio: passare da "He" a "She" non deve rendere il francese più o meno probabile. Se invece confrontassimo "He is the monarch of England," con "Elle est la reine d'Angleterre,", cambierebbe anche la lingua, e la coppia non andrebbe bene.

Il problema di questa definizione è che è scritta interamente in termini di probabilità. Per capire cosa succede quando sommiamo un vettore a $\lambda$ serve invece una caratterizzazione geometrica. Inoltre, trovare in pratica coppie di contesti che soddisfino esattamente le due condizioni è molto difficile. Il Lemma 2.4 risolve il primo problema.

**Lemma 2.4 (relazione tra unembedding ed embedding).** Sia $\bar\lambda_W$ la rappresentazione di embedding di $W$, e siano $\bar\gamma_W$ e $\bar\gamma_Z$ le rappresentazioni di unembedding di $W$ e di un qualsiasi concetto $Z$ causalmente separabile da $W$. Allora:

$$\bar\lambda_W^\top \bar\gamma_W > 0 \qquad \text{e} \qquad \bar\lambda_W^\top \bar\gamma_Z = 0$$

Viceversa, supponiamo che $\bar\lambda_W$ soddisfi queste due condizioni ed esistano $d-1$ concetti $Z_i$, separabili da $W$, tali che $\{\bar\gamma_W\} \cup \{\bar\gamma_{Z_i}\}_{i=1}^{d-1}$ sia una base di $\mathbb{R}^d$. Allora $\bar\lambda_W$ è la rappresentazione di embedding di $W$.

In altre parole, $\bar\lambda_W$ "vede" il proprio concetto (prodotto positivo con $\bar\gamma_W$) ed è "cieco" a tutti gli altri (prodotto nullo con ogni $\bar\gamma_Z$).

La dimostrazione è nel capitolo che hai già negli appunti, quindi qui riprendiamo solo l'idea. Si parte da una coppia $\lambda_0, \lambda_1$ che soddisfa la Definizione 2.3, quindi $\lambda_1 - \lambda_0 = c\,\bar\lambda_W$ con $c > 0$. Restringendo l'output alle quattro parole della tabella (king, roi, queen, reine), le due condizioni diventano condizioni su coppie di parole:

- la prima dice che queen diventa più probabile rispetto a king.
- la seconda dice che roi rispetto a king (e reine rispetto a queen) resta invariata.

A questo punto si applica il Teorema 2.2 a entrambe le coppie. Un logit che cresce diventa $(\lambda_1 - \lambda_0)^\top \bar\gamma_W > 0$, un logit invariato diventa $(\lambda_1 - \lambda_0)^\top \bar\gamma_Z = 0$. Sostituendo $\lambda_1 - \lambda_0 = c\,\bar\lambda_W$ e dividendo per le costanti positive si ottiene la tesi. Tutto il lavoro lo fa il Teorema 2.2, che traduce ogni condizione sulle probabilità in un prodotto scalare.

Per il verso inverso, le condizioni $\bar\lambda^\top \bar\gamma_{Z_i} = 0$ sono $d-1$ equazioni lineari indipendenti in $d$ incognite. Quindi le soluzioni formano una retta passante per l'origine. La condizione $\bar\lambda^\top \bar\gamma_W > 0$ sceglie uno dei due versi. Ne consegue che $\bar\lambda_W$ è unico a meno di una costante positiva, e ogni differenza $\lambda_1 - \lambda_0$ che soddisfa la Definizione 2.3 cade per forza su quella semiretta.

Un'osservazione: $\bar\lambda_W$ in generale **non è parallelo** a $\bar\gamma_W$. Ad esempio, in $d=2$ con $\bar\gamma_W = (1,1)$ e $\bar\gamma_Z = (1,0)$, l'unico vettore (a meno di scala) con prodotto nullo con $\bar\gamma_Z$ e positivo con $\bar\gamma_W$ è $\bar\lambda_W = (0,1)$. Usare $\bar\gamma_W$ al posto di $\bar\lambda_W$ non sarebbe opportuno, in quanto $\bar\gamma_W^\top \bar\gamma_Z = 1 \neq 0$: sommandolo cambierebbe anche la lingua.

Ora il collegamento con lo steering.

**Teorema 2.5 (intervention representation).** Sia $\bar\lambda_W$ la rappresentazione di embedding di $W$. Allora, per ogni concetto $Z$ causalmente separabile da $W$:

$$P\big(Y = Y(W,1) \mid Y \in \{Y(W,0), Y(W,1)\},\ \lambda + c\,\bar\lambda_W\big) \quad \text{è costante in } c \in \mathbb{R}$$

$$P\big(Y = Y(1,Z) \mid Y \in \{Y(0,Z), Y(1,Z)\},\ \lambda + c\,\bar\lambda_W\big) \quad \text{è crescente in } c \in \mathbb{R}$$

Torniamo al nostro esempio. Sommiamo $c\,\bar\lambda_W$ all'embedding di "He is the monarch of England, the". La prima riga dice che il rapporto tra roi e king non cambia (la lingua resta quella). La seconda dice che, a lingua fissata, queen diventa sempre più probabile rispetto a king al crescere di $c$.

La dimostrazione segue direttamente dal lemma. Per la coppia del concetto $W$, il Teorema 2.2 dà:

$$\text{logit} = \alpha\,(\lambda + c\,\bar\lambda_W)^\top \bar\gamma_W = \alpha\,\lambda^\top \bar\gamma_W + \alpha\,c\,\bar\lambda_W^\top \bar\gamma_W$$

dove il secondo termine cresce con $c$, in quanto $\alpha > 0$ e $\bar\lambda_W^\top \bar\gamma_W > 0$. Per la coppia del concetto $Z$ invece:

$$\text{logit} = \alpha'\,(\lambda + c\,\bar\lambda_W)^\top \bar\gamma_Z = \alpha'\,\lambda^\top \bar\gamma_Z + \alpha'\,c \cdot 0$$

che non dipende da $c$. Siccome la sigmoide è crescente, lo stesso vale per le probabilità.

Questo è esattamente quello che si fa con lo **steering**: si somma un vettore all'hidden state per cambiare un concetto nell'output senza toccare gli altri. Il teorema dice che il vettore giusto da sommare è $\bar\lambda_W$, e non $\bar\gamma_W$.

La 2.3 collega quindi la nozione di subspace nello spazio dei contesti alla nozione di intervention. Resta però un problema pratico: come si trova $\bar\lambda_W$ in un modello reale?

Una prima strada è la Definizione 2.3, ma servono coppie di contesti che cambino solo $W$ e lascino esattamente invariati tutti gli altri concetti, e trovarle è molto difficile. Una seconda strada è il lemma letto al contrario, che diventa un modo per costruire $\bar\lambda_W$ senza cercare coppie di contesti. Il lemma dice che $\bar\lambda_W$ è il vettore con:

$$\bar\lambda_W^\top \bar\gamma_W > 0 \qquad \text{e} \qquad \bar\lambda_W^\top \bar\gamma_{Z_i} = 0 \ \text{ per ogni } Z_i$$

dove ogni concetto $Z_i$ separabile da $W$ fornisce un'equazione lineare. Il problema è quante equazioni servono per individuare un'unica direzione. $\bar\lambda_W$ ha $d$ componenti, e ogni equazione $\bar\lambda^\top \bar\gamma_{Z_i} = 0$ toglie una dimensione:

- **con 1 equazione** le soluzioni formano un iperpiano, di dimensione $d-1$.
- **con 2 equazioni** formano un sottospazio di dimensione $d-2$.
- **con $d-1$ equazioni indipendenti** resta una retta, ovvero una sola direzione. La condizione $> 0$ sceglie poi il verso.

Vediamolo con un modello giocattolo con $d = 3$. Supponiamo di conoscere le rappresentazioni di unembedding di tre concetti:

- $W$ = male⇒female: $\bar\gamma_W = (1, 1, 0)$
- $Z_1$ = English⇒French: $\bar\gamma_{Z_1} = (1, 0, 0)$
- $Z_2$ = singolare⇒plurale: $\bar\gamma_{Z_2} = (0, 1, 1)$

Vogliamo il vettore di steering $\bar\lambda_W = (a, b, c)$: sommato al contesto, deve rendere "queen" più probabile di "king" senza cambiare lingua e numero.

Come primo tentativo usiamo solo $Z_1$, imponendo le condizioni del lemma su $W$ e sulla lingua:

$$\bar\lambda_W^\top \bar\gamma_{Z_1} = a = 0 \qquad \bar\lambda_W^\top \bar\gamma_W = a + b > 0$$

quindi $a = 0$ e $b > 0$, mentre $c$ può essere qualsiasi. I candidati sono infiniti, ad esempio $(0, 1, 0)$, $(0, 1, 5)$ e $(0, 1, -1)$. Tutti rispettano le due condizioni, e con queste sole informazioni non abbiamo modo di sceglierne uno. Il problema si vede quando controlliamo il concetto che non abbiamo vincolato, ovvero il numero, calcolando $\bar\lambda^\top \bar\gamma_{Z_2} = b + c$:

| candidato | genere ($a+b$) | lingua ($a$) | numero ($b+c$) |
|---|---|---|---|
| $(0, 1, 0)$ | 1 ✓ | 0 ✓ | 1 ✗ |
| $(0, 1, 5)$ | 1 ✓ | 0 ✓ | 6 ✗ |
| $(0, 1, -1)$ | 1 ✓ | 0 ✓ | 0 ✓ |

Se usassimo $(0, 1, 0)$, partendo da un contesto che porta a "king" non otterremmo solo "queen": spingeremmo anche verso "queens", perché il plurale diventa più probabile. Con $(0, 1, 5)$ l'effetto sul plurale sarebbe ancora più forte.

Aggiungiamo quindi $Z_2$. Ora le condizioni sono tre:

$$a = 0, \qquad b + c = 0, \qquad a + b > 0$$

quindi $b > 0$ e $c = -b$. Resta una sola direzione, $\bar\lambda_W = (0, 1, -1)$ a meno di una costante positiva, che cambia il genere e lascia invariati lingua e numero. In $d = 3$ sono serviti quindi $3 - 1 = 2$ concetti oltre a $W$ (allo stesso modo, nell'esempio in $d=2$ ne bastava uno). Ogni concetto aggiunge un'equazione e toglie un grado di libertà, finché non resta una sola direzione. Quando un concetto manca, il vettore trovato rischia di modificare proprio quel concetto senza che ce ne accorgiamo.

In LLaMA-2 (7B) vale lo stesso ragionamento, ma l'hidden state ha dimensione $d = 4096$. Servirebbero quindi $4096 - 1 = 4095$ concetti separabili da $W$, i cui $\bar\gamma_{Z_i}$, insieme a $\bar\gamma_W$, formino una base di $\mathbb{R}^{4096}$. Ogni $\bar\gamma_{Z_i}$ va stimato da coppie controfattuali di parole, e negli esperimenti il paper ha 27 concetti in tutto. Anche usandoli tutti, rimarrebbe uno spazio di soluzioni di dimensione superiore a 4000, con moltissime direzioni che cambiano concetti che non abbiamo considerato.

Entrambe le strade sono quindi impraticabili. La sezione 3 risolve il problema con il **prodotto interno causale**. Se scegliamo un prodotto interno in cui concetti separabili sono ortogonali, $\bar\lambda_W$ si ottiene direttamente da $\bar\gamma_W$ (in pratica $\bar\lambda_W \propto \text{Cov}(\gamma)^{-1}\bar\gamma_W$). Questa formula usa solo le coppie di parole del concetto $W$ e la matrice di unembedding, senza bisogno di conoscere gli altri 4095 concetti.

## Sezione 3: Inner Product for Language Model Representations

### Ripasso di algebra lineare

Prima di entrare nella sezione 3 conviene fissare tre strumenti di algebra lineare, perché il paper li usa senza introdurli.

**Prodotto interno come matrice.** Un prodotto interno è una funzione che prende due vettori $u, v \in \mathbb{R}^d$ e restituisce un solo numero, quindi un elemento di $\mathbb{R}$ (non di $\mathbb{R}^d$). Il caso più noto è il prodotto scalare euclideo, $u^\top v = u_1 v_1 + \dots + u_d v_d$. In generale, un prodotto interno deve rispettare tre proprietà:

- **Bilinearità**: è lineare in ciascuno dei due argomenti, ovvero si comporta come una moltiplicazione rispetto a somme e costanti. Tenendo fisso il secondo vettore vale $\langle a\,u + b\,w,\ v \rangle = a\,\langle u, v \rangle + b\,\langle w, v \rangle$, e lo stesso vale per il secondo argomento. In pratica si può "distribuire" il prodotto sulle somme e portare fuori le costanti, come si fa con $(a+b)(c+d) = ac + ad + bc + bd$.
- **Simmetria**: $\langle u, v \rangle = \langle v, u \rangle$.
- **Positività**: $\langle u, u \rangle > 0$ per ogni $u \neq 0$, ovvero ogni vettore non nullo ha lunghezza positiva.

In $\mathbb{R}^d$ ogni prodotto interno si può scrivere come:

$$\langle u, v \rangle_M = u^\top M v$$

dove $M \in \mathbb{R}^{d \times d}$ è una matrice simmetrica e definita positiva. Per capire perché, serve la **base canonica**, ovvero i $d$ vettori che hanno un 1 in una posizione e 0 altrove. In $\mathbb{R}^2$ sono $e_1 = (1, 0)$ ed $e_2 = (0, 1)$. Ogni vettore si scrive come combinazione di questi, ad esempio $u = (3, 5) = 3\,e_1 + 5\,e_2$, dove i coefficienti sono semplicemente le componenti del vettore. Gli $e_i$ sono quindi i "mattoncini" con cui si costruisce qualsiasi vettore.

Restiamo in $\mathbb{R}^2$. Scriviamo $u = u_1 e_1 + u_2 e_2$ e $v = v_1 e_1 + v_2 e_2$, poi distribuiamo con la bilinearità:

$$\langle u, v \rangle = u_1 v_1 \langle e_1, e_1 \rangle + u_1 v_2 \langle e_1, e_2 \rangle + u_2 v_1 \langle e_2, e_1 \rangle + u_2 v_2 \langle e_2, e_2 \rangle$$

Il prodotto di due vettori qualsiasi dipende quindi solo da quattro numeri, ovvero i prodotti tra i vettori della base. Se li conosciamo, conosciamo il prodotto interno su tutto lo spazio. Raccogliamo questi numeri in una matrice:

$$M = \begin{pmatrix} \langle e_1, e_1 \rangle & \langle e_1, e_2 \rangle \\ \langle e_2, e_1 \rangle & \langle e_2, e_2 \rangle \end{pmatrix}, \qquad \text{ovvero } M_{ij} = \langle e_i, e_j \rangle$$

dove la somma dei quattro termini di prima è esattamente quello che si ottiene facendo il conto $u^\top M v$ (riga per matrice per colonna). Quindi $u^\top M v$ non è una definizione presa dal nulla: è il prodotto interno scritto in forma compatta. In $\mathbb{R}^d$ il ragionamento è lo stesso, solo con $d^2$ numeri invece di 4.

Le altre due proprietà si traducono in condizioni su $M$. La simmetria dà $\langle e_1, e_2 \rangle = \langle e_2, e_1 \rangle$, quindi $M_{12} = M_{21}$ e $M$ è simmetrica. La positività dà $u^\top M u > 0$ per ogni $u \neq 0$, che è proprio la definizione di matrice definita positiva. Ad esempio, una matrice con $-1$ al posto di $\langle e_2, e_2 \rangle$ non andrebbe bene, perché $e_2$ avrebbe "lunghezza al quadrato" pari a $-1$.

Con il prodotto euclideo i vettori della base sono lunghi 1 e sono ortogonali, quindi $\langle e_1, e_1 \rangle = \langle e_2, e_2 \rangle = 1$ e $\langle e_1, e_2 \rangle = 0$. Ne consegue che $M = I$, e $u^\top I v = u_1 v_1 + u_2 v_2$ è il solito prodotto scalare. Possiamo però scegliere altri valori, ad esempio $\langle e_2, e_2 \rangle = 4$. In questo caso otteniamo la **matrice diagonale**:

$$M = \text{diag}(1, 4) = \begin{pmatrix} 1 & 0 \\ 0 & 4 \end{pmatrix}$$

dove in generale $\text{diag}(a_1, \dots, a_d)$ indica la matrice con $a_1, \dots, a_d$ sulla diagonale e zero altrove (la matrice identità, ad esempio, è $\text{diag}(1, \dots, 1)$). Il prodotto interno corrispondente è:

$$u^\top M v = u_1 v_1 + 4\,u_2 v_2$$

dove la prima coordinata conta come nel prodotto euclideo e la seconda conta quattro volte tanto. È un prodotto interno legittimo, in quanto la matrice è simmetrica e, avendo valori positivi sulla diagonale, definita positiva. Quindi scegliere un prodotto interno su $\mathbb{R}^d$ vuol dire scegliere una matrice $M$ simmetrica e definita positiva, e viceversa. Per questo nella sezione 3 la domanda "quale prodotto interno usare?" diventa "quale matrice $M$ usare?".

Il punto che interessa a noi è che lunghezze e angoli dipendono da $M$. Prendiamo $u = (1, 1)$ e $v = (1, -1)$. Con il prodotto euclideo $u^\top v = 0$, quindi i due vettori sono ortogonali. Con $M = \text{diag}(1, 4)$ invece:

$$u^\top M v = 1 \cdot 1 \cdot 1 + 1 \cdot 4 \cdot (-1) = -3$$

quindi non lo sono più. L'ortogonalità quindi non è una proprietà dei soli vettori, ma dei vettori insieme al prodotto interno scelto. È esattamente il problema da cui parte la sezione 3.

**Covarianza di una trasformazione lineare.** Sia $\gamma$ un vettore aleatorio in $\mathbb{R}^d$ con media $\mu$. La sua matrice di covarianza è:

$$\text{Cov}(\gamma) = \mathbb{E}\big[(\gamma - \mu)(\gamma - \mu)^\top\big]$$

dove l'elemento $(i,j)$ è la covarianza tra la componente $i$ e la componente $j$. Se trasformiamo $\gamma$ con una matrice $A$ e sommiamo una costante $\beta$, la covarianza diventa:

$$\text{Cov}(A\gamma + \beta) = A\,\text{Cov}(\gamma)\,A^\top$$

La dimostrazione è immediata. La media di $A\gamma + \beta$ è $A\mu + \beta$, quindi la costante $\beta$ si cancella e resta $A(\gamma - \mu)$. Sostituendo nella definizione, $A$ esce a sinistra e $A^\top$ a destra.

Nella sezione 3, $\gamma$ sarà l'unembedding di una parola estratta uniformemente dal vocabolario. Quindi $\text{Cov}(\gamma)$ è semplicemente la covarianza empirica delle righe della matrice di output (32K vettori, nel caso di LLaMA-2), ovvero una quantità che possiamo calcolare direttamente.

**Radice quadrata di una matrice.** Una matrice $M$ simmetrica e definita positiva si può diagonalizzare come $M = Q \Lambda Q^\top$, con $Q$ ortogonale e autovalori in $\Lambda$ tutti positivi. La sua radice quadrata è:

$$M^{1/2} = Q \Lambda^{1/2} Q^\top$$

dove $\Lambda^{1/2}$ contiene le radici degli autovalori. È l'unica matrice simmetrica e definita positiva tale che $M^{1/2} M^{1/2} = M$. Ad esempio, per $M = \text{diag}(1, 4)$ si ha $M^{1/2} = \text{diag}(1, 2)$.

Il motivo per cui ci serve è il seguente:

$$u^\top M v = u^\top M^{1/2} M^{1/2} v = (M^{1/2} u)^\top (M^{1/2} v)$$

In altre parole, **ogni prodotto interno è il prodotto euclideo dopo aver trasformato i vettori con $M^{1/2}$**. Geometricamente, $\text{diag}(1, 2)$ stira lo spazio lungo il secondo asse, raddoppiando la seconda coordinata. Nell'esempio di prima $u$ diventa $(1, 2)$ e $v$ diventa $(1, -2)$, il cui prodotto euclideo è $1 - 4 = -3$: è per questo che $u$ e $v$ smettono di essere ortogonali. Questa osservazione tornerà alla fine della sezione.

Un caso particolare che conviene avere in mente è $M = \text{Cov}(\gamma)^{-1}$. Applicando la regola della covarianza:

$$\text{Cov}\big(\text{Cov}(\gamma)^{-1/2}\gamma\big) = \text{Cov}(\gamma)^{-1/2}\,\text{Cov}(\gamma)\,\text{Cov}(\gamma)^{-1/2} = I$$

quindi i vettori trasformati hanno covarianza identità. Questa operazione si chiama **whitening**, ed è proprio quello che farà il paper alla fine della sezione.

### Preliminari

**Lo spazio delle differenze $\bar\Gamma$.** Il paper non definisce il prodotto interno su $\Gamma$, ma su $\bar\Gamma$, ovvero lo spazio delle differenze $\gamma(y) - \gamma(y')$. Il motivo è che $\Gamma$ è solo uno **spazio affine** (nota 1 del paper), ovvero uno spazio di punti in cui non c'è un'origine privilegiata. Ha senso fare la differenza tra due punti, che dà un vettore, ma non ha senso dire dove si trova un punto "in assoluto".

L'esempio classico è quello delle date sul calendario. Possiamo dire che tra il 3 marzo e il 10 marzo passano 7 giorni, quindi la differenza tra due date è ben definita. Non ha senso invece sommare due date ("3 marzo + 10 marzo"), e anche il "giorno zero" è una convenzione arbitraria: se lo spostiamo, tutte le differenze restano uguali. In uno spazio vettoriale invece l'origine è fissata, quindi possiamo sommare due vettori, moltiplicarli per uno scalare e parlare della lunghezza di un singolo vettore.

Per gli unembedding succede la stessa cosa: la softmax non cambia se sommiamo la stessa costante $\beta$ a tutti i $\gamma(y)$. Infatti:

$$\exp\big(\lambda^\top (\gamma(y) + \beta)\big) = \exp\big(\lambda^\top \gamma(y)\big) \cdot \exp\big(\lambda^\top \beta\big)$$

dove il secondo fattore è lo stesso per tutte le parole $y$, quindi si cancella con il denominatore della softmax. Ne consegue che un modello con unembedding $\gamma(y)$ e uno con $\gamma(y) + \beta$ producono esattamente le stesse probabilità. Spostare l'origine di $\Gamma$ dà quindi un modello identico.

La posizione di una singola parola, come $\gamma(\text{king})$, è quindi arbitraria, perché dipende da dove abbiamo messo l'origine. Le differenze invece sono ben definite, in quanto $\beta$ si cancella:

$$\big(\gamma(y) + \beta\big) - \big(\gamma(y') + \beta\big) = \gamma(y) - \gamma(y')$$

È per questo che nella sezione 2 si lavorava sempre con differenze, come $\gamma(\text{queen}) - \gamma(\text{king})$. Lo spazio $\bar\Gamma$ di queste differenze è un vero spazio vettoriale di dimensione $d$. È lì che ha senso definire un prodotto interno, perché per parlare di lunghezze e angoli serve un'origine.

**Rappresentazione canonica.** Dalla Definizione 2.1 sappiamo che $\bar\gamma_W$ è unico solo a meno di una costante positiva: è una direzione, non un vettore. Per confrontare due concetti con un prodotto interno questo è un problema, perché $\langle \bar\gamma_W, \bar\gamma_Z \rangle$ dipenderebbe dalla lunghezza scelta arbitrariamente per i due vettori.

Si risolve fissando la lunghezza. Una volta scelto un prodotto interno, la **rappresentazione canonica** di $W$ è l'unico elemento del cono con norma 1:

$$\langle \bar\gamma_W, \bar\gamma_W \rangle = 1$$

In questo modo $\langle \bar\gamma_W, \bar\gamma_Z \rangle$ diventa una specie di cosine similarity tra i due concetti.

Attenzione però: la norma dipende dal prodotto interno, quindi anche la rappresentazione canonica dipende da questa scelta. Riprendendo l'esempio della 2.2, con il cono di $v = (1,1)$:

- con il prodotto euclideo, $\langle v, v \rangle = 2$, quindi la rappresentazione canonica è $v/\sqrt{2}$.
- con $M = \text{diag}(1, 4)$, $\langle v, v \rangle = 1 + 4 = 5$, quindi è $v/\sqrt{5}$.

Fino a qui abbiamo dato per scontato di avere un prodotto interno. La prossima parte mostra che sceglierlo non è affatto banale.

### Non identificabilità del prodotto interno

Si potrebbe pensare di usare semplicemente il prodotto scalare euclideo. Ad esempio, per capire se genere e lingua sono legati potremmo calcolare la cosine similarity tra $\bar\gamma_W$ e $\bar\gamma_Z$. Per modificare un concetto potremmo invece proiettare su una direzione. L'idea è naturale, ma presuppone che il training del modello abbia "scelto" un prodotto interno privilegiato. Il paper mostra che non è così.

Consideriamo la trasformazione (3.1):

$$g(y) \leftarrow A\gamma(y) + \beta, \qquad l(x) \leftarrow A^{-\top}\lambda(x)$$

dove $A \in \mathbb{R}^{d \times d}$ è una qualsiasi matrice invertibile e $\beta \in \mathbb{R}^d$ una costante. In pratica si trasformano gli unembedding con $A$ e gli embedding con l'inversa trasposta. Verifichiamo che le probabilità non cambiano calcolando il nuovo logit:

$$l(x)^\top g(y) = \big(A^{-\top}\lambda(x)\big)^\top \big(A\gamma(y) + \beta\big) = \lambda(x)^\top A^{-1} A\,\gamma(y) + \lambda(x)^\top A^{-1}\beta$$

dove $A^{-1}A = I$, quindi il primo termine è proprio il logit originale $\lambda(x)^\top \gamma(y)$. Il secondo termine non dipende da $y$, quindi si cancella nella softmax per lo stesso motivo visto nei preliminari. Ne consegue che:

$$P(y \mid x) \text{ calcolata con } (\lambda, \gamma) = P(y \mid x) \text{ calcolata con } (l, g), \qquad \forall x, y$$

La funzione obiettivo del training (la cross-entropy) dipende dalle rappresentazioni solo attraverso queste probabilità. Quindi i due modelli hanno la stessa loss e producono gli stessi output, e il training non ha alcun modo di preferire uno all'altro. In altre parole, $\gamma$ è identificato (al massimo) **solo a meno di una trasformazione affine invertibile**. In pratica, la matrice $A$ potrebbe essere assorbita nella matrice di output e $A^{-\top}$ nell'ultimo layer del transformer, senza cambiare nulla del comportamento del modello.

Vediamo cosa succede alle rappresentazioni dei concetti. Le differenze tra unembedding diventano:

$$g(y_1) - g(y_0) = A\big(\gamma(y_1) - \gamma(y_0)\big)$$

quindi $\beta$ sparisce e $\bar\gamma_W$ diventa $A\bar\gamma_W$. La buona notizia è che la nozione di rappresentazione lineare sopravvive: se le differenze stavano tutte nel cono di $\bar\gamma_W$, dopo la trasformazione stanno tutte nel cono di $A\bar\gamma_W$, perché $A(\alpha\bar\gamma_W) = \alpha A\bar\gamma_W$. La cattiva notizia è che il prodotto interno non sopravvive. Con il prodotto euclideo:

$$(A\bar\gamma_W)^\top (A\bar\gamma_Z) = \bar\gamma_W^\top A^\top A\,\bar\gamma_Z$$

che coincide con $\bar\gamma_W^\top \bar\gamma_Z$ solo se $A^\top A = I$, ovvero se $A$ è una rotazione o una riflessione. Per una $A$ generica, invece, il valore cambia.

Torniamo al nostro esempio con $d = 2$. Supponiamo che genere e lingua abbiano $\bar\gamma_W = (1, 0)$ e $\bar\gamma_Z = (0, 1)$, quindi siano ortogonali con il prodotto euclideo. Prendiamo:

$$A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$$

dove $A$ è invertibile (determinante 1). Le nuove rappresentazioni sono $A\bar\gamma_W = (1, 0)$ e $A\bar\gamma_Z = (1, 1)$, e il loro prodotto euclideo vale 1. L'angolo tra i due concetti è passato da 90° a 45°. Il modello però è identico: stesse probabilità per ogni contesto e ogni parola. Con un'altra $A$ potremmo fare il contrario, cioè rendere ortogonali due concetti che prima non lo erano.

Quindi dire "nel modello genere e lingua sono ortogonali nel senso euclideo" non dice niente sul modello, perché dipende da una scelta arbitraria che il training non fissa. Per questo non c'è motivo di aspettarsi che le operazioni basate sul prodotto euclideo (cosine similarity, proiezioni) abbiano un significato semantico.

C'è un modo equivalente di vedere la cosa, che collega questo risultato al ripasso iniziale. Cambiare rappresentazione con $A$ e poi usare il prodotto euclideo è lo stesso che tenere la rappresentazione originale e usare il prodotto interno con matrice $M = A^\top A$. Viceversa, ogni prodotto interno $M$ si ottiene così, con $A = M^{1/2}$. Quindi **scegliere una rappresentazione e scegliere un prodotto interno sono la stessa cosa**, e il training non fissa né l'una né l'altro.

Serve quindi un principio esterno al training per scegliere il prodotto interno. È quello che fa la 3.1 con il prodotto interno causale: si sceglie l'$M$ in cui i concetti causalmente separabili risultano ortogonali.

### Prodotto interno causale (Definizione 3.1)

Il training non fissa nessun prodotto interno, quindi serve un principio esterno. Il paper sceglie quello più naturale alla luce della sezione 2: due concetti causalmente separabili devono essere ortogonali. Ad esempio, male⇒female ed English⇒French possono variare in modo indipendente (la tabella king/roi/queen/reine), quindi le loro rappresentazioni devono essere perpendicolari.

**Definizione 3.1.** Un prodotto interno $\langle \cdot, \cdot \rangle_C$ su $\bar\Gamma$ si dice **causale** se:

$$\langle \bar\gamma_W, \bar\gamma_Z \rangle_C = 0$$

per ogni coppia di concetti $W$ e $Z$ causalmente separabili.

L'idea non arriva dal nulla. Il Lemma 2.4 chiedeva già un prodotto nullo tra concetti separabili, $\bar\lambda_W^\top \bar\gamma_Z = 0$, ma mescolava i due spazi (un embedding e un unembedding). La Definizione 3.1 chiede la stessa cosa dentro un solo spazio.

### Teorema 3.2 (unificazione delle rappresentazioni)

**Teorema 3.2.** Supponiamo che per ogni concetto $W$ esistano $d-1$ concetti $Z_i$ separabili da $W$, tali che $\{\bar\gamma_W\} \cup \{\bar\gamma_{Z_i}\}_{i=1}^{d-1}$ sia una base di $\mathbb{R}^d$. Se $\langle \cdot, \cdot \rangle_C$ è causale, allora l'isomorfismo di Riesz manda ogni rappresentazione di unembedding nella corrispondente rappresentazione di embedding:

$$\langle \bar\gamma_W, \cdot \rangle_C = \bar\lambda_W^\top$$

Per capire l'enunciato serve l'isomorfismo di Riesz. Un **funzionale lineare** è una funzione lineare che prende un vettore e restituisce un numero. Gli embedding sono proprio questo: ogni $\lambda$ definisce il funzionale $\bar\gamma \mapsto \lambda^\top \bar\gamma$, ovvero una variazione di logit. Possiamo quindi pensare gli embedding come vettori riga che agiscono sui vettori colonna degli unembedding.

Fissato un prodotto interno, ogni vettore $\bar\gamma$ definisce un funzionale: "fai il prodotto interno con $\bar\gamma$". Il teorema di Riesz dice che vale anche il contrario, ovvero ogni funzionale si ottiene così da un unico vettore. Nel caso euclideo l'isomorfismo è semplicemente la trasposta. Con una matrice $M$ invece:

$$\langle \bar\gamma_W, \cdot \rangle_C = \bar\gamma_W^\top M = (M\bar\gamma_W)^\top$$

quindi il teorema dice semplicemente che $\bar\lambda_W = M\bar\gamma_W$.

La dimostrazione (Appendice B.4) è molto banale: basta verificare le due condizioni del Lemma 2.4. Il prodotto $\langle \bar\gamma_W, \bar\gamma_W \rangle_C$ è positivo per definizione di prodotto interno, mentre $\langle \bar\gamma_W, \bar\gamma_Z \rangle_C = 0$ per la Definizione 3.1. Grazie all'ipotesi sulla base vale il verso inverso del lemma, e la tesi segue. In altre parole, il prodotto interno causale è costruito apposta per rendere vere le condizioni del lemma.

Torniamo all'esempio in $d = 2$ della 2.3, con $\bar\gamma_W = (1, 1)$ (genere), $\bar\gamma_Z = (1, 0)$ (lingua) e $\bar\lambda_W = (0, 1)$. Il prodotto euclideo non è causale, in quanto $\bar\gamma_W^\top \bar\gamma_Z = 1$. La trasposta darebbe infatti $(1, 1)$, che cambia anche la lingua. Consideriamo invece:

$$M = \begin{pmatrix} 1 & -1 \\ -1 & 2 \end{pmatrix}$$

dove $M$ è simmetrica e definita positiva (determinante 1, traccia positiva). Questo prodotto interno è causale, in quanto $\bar\gamma_W^\top M \bar\gamma_Z = (0, 1) \cdot (1, 0) = 0$. Applicando il teorema:

$$\bar\lambda_W = M\bar\gamma_W = (0, 1)$$

ovvero proprio il vettore trovato nella 2.3.

Abbiamo quindi una terza strada per costruire $\bar\lambda_W$: stimare $\bar\gamma_W$ dalle coppie di parole e moltiplicarlo per $M$. Il problema è che, per trovare $M$ con la sola Definizione 3.1, servirebbero di nuovo moltissimi concetti. La 3.2 risolve proprio questo.

### Assunzione 3.3

L'idea della 3.2 è collegare la separabilità causale alla dipendenza statistica tra parole, che invece si può misurare. Se estraiamo una parola a caso e scopriamo che è francese, questo non ci dice quasi niente sul fatto che si riferisca a un uomo o a una donna.

**Assunzione 3.3.** Siano $W$ e $Z$ causalmente separabili e sia $\gamma$ l'unembedding di una parola estratta uniformemente dal vocabolario. Allora $\bar\lambda_W^\top \gamma$ e $\bar\lambda_Z^\top \gamma$ sono indipendenti.

Il numero $\bar\lambda_W^\top \gamma$ si può leggere come il "punteggio" della parola sul concetto $W$: "queen" avrà un punteggio alto sul genere, "king" uno basso. L'assunzione dice quindi che il punteggio sul genere e quello sulla lingua non si influenzano. Ci sono due precisazioni (note 2 e 3 del paper):

- **Campionamento**: le parole vengono estratte dal vocabolario, non dal testo naturale. Nei testi possono esserci correlazioni non causali, mentre nel vocabolario ogni parola conta una volta.
- **Incorrelazione**: per la dimostrazione basta che i due punteggi siano incorrelati, una condizione più debole dell'indipendenza. L'Appendice D.6 verifica che il prodotto interno stimato la rispetti.

### Teorema 3.4 (forma esplicita)

**Teorema 3.4.** Supponiamo che esista un prodotto interno causale $\langle \bar\gamma, \bar\gamma' \rangle_C = \bar\gamma^\top M \bar\gamma'$. Supponiamo inoltre che esistano $d$ concetti $W_1, \dots, W_d$, separabili a due a due, le cui rappresentazioni canoniche $G = [\bar\gamma_{W_1}, \dots, \bar\gamma_{W_d}]$ formino una base. Allora, sotto l'Assunzione 3.3:

$$M^{-1} = GG^\top \qquad \text{e} \qquad G^\top \text{Cov}(\gamma)^{-1} G = D$$

dove $D$ è una matrice diagonale con elementi positivi.

La dimostrazione (Appendice B.5) si divide in due parti. Nella prima, l'elemento $(i, j)$ di $G^\top M G$ è $\langle \bar\gamma_{W_i}, \bar\gamma_{W_j} \rangle_C$. Questo vale 0 per $i \neq j$ (prodotto causale) e 1 per $i = j$ (rappresentazioni canoniche). Ne consegue che:

$$G^\top M G = I \quad \Rightarrow \quad M = G^{-\top} G^{-1}$$

ovvero $M^{-1} = GG^\top$. In pratica $G$ è una base ortonormale rispetto al prodotto interno causale.

Nella seconda parte si utilizza l'assunzione. Per il Teorema 3.2 si ha $\bar\lambda_{W_i} = M\bar\gamma_{W_i}$, e i punteggi di due concetti diversi sono incorrelati. Ricordando che $\text{Cov}(a^\top\gamma, b^\top\gamma) = a^\top \text{Cov}(\gamma)\, b$:

$$0 = \bar\gamma_{W_i}^\top M\,\text{Cov}(\gamma)\,M\,\bar\gamma_{W_j} \qquad (i \neq j)$$

quindi $G^\top M\,\text{Cov}(\gamma)\,M G$ è diagonale. Sulla diagonale ci sono le varianze dei punteggi, positive, e la chiamiamo $D^{-1}$. Sostituendo $G^\top M = G^{-1}$ e $MG = G^{-\top}$ (dalla prima parte):

$$G^{-1}\,\text{Cov}(\gamma)\,G^{-\top} = D^{-1} \quad \Rightarrow \quad \text{Cov}(\gamma)^{-1} = G^{-\top} D\, G^{-1}$$

da cui segue $G^\top \text{Cov}(\gamma)^{-1} G = D$. Confrontando le due parti si ottiene:

$$M = G^{-\top}\, I\, G^{-1}, \qquad \text{Cov}(\gamma)^{-1} = G^{-\top}\, D\, G^{-1}$$

quindi $M$ e $\text{Cov}(\gamma)^{-1}$ coincidono a meno di un riscalamento lungo le direzioni dei concetti.

### Gradi di libertà e scelta di D

Una matrice simmetrica $d \times d$ ha $d(d-1)/2 + d$ gradi di libertà. L'ortogonalità tra $d$ concetti impone un vincolo per ogni coppia, quindi $d(d-1)/2$ vincoli. Restano $d$ gradi di libertà, ovvero proprio gli elementi di $D$.

L'intuizione è che l'ortogonalità dice quali direzioni sono perpendicolari, ma non quanto sono "lunghe". Lo avevamo visto nel ripasso: con $I$ e con $\text{diag}(1, 4)$ i vettori $e_1$ ed $e_2$ sono ortogonali in entrambi i casi, ma la seconda direzione viene misurata in modo diverso.

Il paper non ha un criterio per scegliere $D$, quindi utilizza la scelta più semplice, $D = I$. In questo caso $M = \text{Cov}(\gamma)^{-1}$ e si ottiene la forma chiusa (3.3):

$$\langle \bar\gamma, \bar\gamma' \rangle_C = \bar\gamma^\top \text{Cov}(\gamma)^{-1} \bar\gamma'$$

dove $\text{Cov}(\gamma)$ è la covarianza delle 32K righe della matrice di output. Scegliere $D = I$ vuol dire chiedere che tutti i punteggi abbiano varianza 1 sul vocabolario, ovvero che ogni concetto abbia la stessa scala.

Anche se il prodotto interno causale non è unico, il teorema ne esclude la maggior parte. Il prodotto euclideo, ad esempio, richiederebbe direzioni dei concetti ortonormali in senso euclideo ($GG^\top = I$) e anche autovettori della covarianza. Non c'è motivo che succeda in un modello reale (l'Appendice D.2 confronta i due prodotti su LLaMA-2 e Gemma).

Mettendo insieme il Teorema 3.2 e la scelta $D = I$ si ottiene la formula anticipata alla fine della 2.3:

$$\bar\lambda_W = \text{Cov}(\gamma)^{-1}\,\bar\gamma_W$$

dove $\bar\gamma_W$ si stima dalle coppie di parole e $\text{Cov}(\gamma)$ dalla matrice di output. Non servono coppie di contesti né gli altri 4095 concetti, ed è la formula (4.1) utilizzata negli esperimenti.

### Rappresentazioni unificate

Abbiamo visto che scegliere un prodotto interno equivale a scegliere una rappresentazione. Prendiamo allora $A = M^{1/2}$ nella trasformazione (3.1), quindi $g = M^{1/2}\gamma$ e $l = M^{-1/2}\lambda$. La rappresentazione di unembedding diventa $\bar g_W = M^{1/2}\bar\gamma_W$, mentre quella di embedding diventa:

$$\bar l_W = M^{-1/2}\bar\lambda_W = M^{-1/2} M\,\bar\gamma_W = M^{1/2}\bar\gamma_W = \bar g_W$$

Nello spazio trasformato le due rappresentazioni **coincidono**: il vettore che misura un concetto (probing) e quello che lo modifica (steering) sono lo stesso. Inoltre, per quanto visto nel ripasso, $\langle \bar\gamma, \bar\gamma' \rangle_C = \bar g^\top \bar g'$. Quindi in questo spazio il prodotto euclideo è proprio quello causale (Figura 1 del paper).

Con $M = \text{Cov}(\gamma)^{-1}$ la trasformazione è il **whitening** visto nel ripasso. In pratica basta fare whitening degli unembedding, e da lì in poi cosine similarity, proiezioni e somme di vettori hanno un significato causale.

### Verifica finale

La sezione 2 associava le tre nozioni dell'introduzione a definizioni e teoremi precisi:

- **Subspace**: Definizione 2.1 ($\bar\gamma_W$) e Definizione 2.3 ($\bar\lambda_W$).
- **Measurement**: Teorema 2.2, dove $\bar\gamma_W$ fa da probe lineare.
- **Intervention**: Teorema 2.5, dove $\bar\lambda_W$ fa da vettore di steering.

La sezione 3 collega le due rappresentazioni con $\bar\lambda_W = M\bar\gamma_W$, e nello spazio trasformato le fa coincidere. Restano però alcune ipotesi:

- **Esistenza del prodotto interno causale**: viene assunta, non dimostrata.
- **Base di concetti separabili**: deve esistere, ma non serve conoscerla, perché la formula finale utilizza solo $\bar\gamma_W$ e $\text{Cov}(\gamma)$.
- **Assunzione 3.3**: è un'ipotesi sui dati, verificata a posteriori nell'Appendice D.6.
- **$D = I$**: è una scelta arbitraria tra i prodotti interni causali possibili.

Gli esperimenti della sezione 4 servono appunto a controllare che, con queste ipotesi, le predizioni reggano su un modello reale.

[^1]: Nel testo il paper dice che a essere logit-lineare è la probabilità di "king". Con la convenzione male⇒female, però, $Y(1)$ è "queen". Per "king" vale la stessa formula con il segno invertito.
