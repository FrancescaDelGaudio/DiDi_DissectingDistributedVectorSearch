# Studio preliminare: le query e i neighbor sono concentrati?

6 novembre 2026. È la sezione 6 della proposta di secondment.

**In sintesi**
1. **Nei log reali lo skew è forte.** Su ORCAS (click veri di Bing), un primo livello T1 con il **10% dei documenti** copre il **77% dei click** di query mai viste, e risponde completamente al **69% delle query**. Con una distribuzione uniforme si otterrebbe il 10%.
2. **Nei benchmark ANN lo skew sulle query non c'è, per costruzione.** Le query sono campionate e deduplicate. Lì i neighbor esatti si ripetono poco: quasi nulla su DEEP, in modo moderato su TTI e MSMARCO (query fuori distribuzione).
3. **I neighbor sono raggruppati nello spazio.** Su MSMARCO-WS 10M i 10 NN di una query stanno in media in **5.7 celle k-means su 1024**, contro 9.9 per 10 documenti a caso. Il 10% dello spazio, scelto dalle query di train, contiene il **25%** dei NN delle query di test. È però troppo poco per rispondere alle query solo con T1, se il traffico è uniforme.

**Cosa ne segue per l'idea:** il guadagno deve venire dallo skew del traffico reale, che i benchmark eliminano. La località spaziale è un vantaggio in più: permette di costruire i livelli per cluster e di generalizzare a query nuove.

---

## 1. ORCAS: frequenza dei documenti nei click reali

**Dati.** ORCAS contiene 18.8M coppie (query, documento cliccato) dai log di Bing, sul corpus MS MARCO Document (3.21M documenti). Ci sono 10.4M query, tutte distinte, e non ci sono timestamp, utenti o numero di click.
Uso il documento cliccato come "neighbor rilevante" della query.

![ORCAS](results/fig_orcas.png)

**Popolarità dei documenti** (grafico a sinistra, assi log-log):
- Solo il 44% del corpus (1.42M documenti) riceve almeno un click; il 56% non viene mai cliccato.
- La curva rango–frequenza è quasi una legge di potenza. Il documento più popolare (facebook.com) è cliccato da 11.310 query distinte; 355k documenti da una sola.
- Quota di coppie query–documento coperta dai documenti più popolari:

  | top % del corpus | 0.1% | 1% | 5% | 10% | 20% |
  |---|---|---|---|---|---|
  | quota dei click | 12% | 36% | 65% | 79% | 92% |

  L'indice di Gini sull'intero corpus è 0.88.

**Generalizzazione a query nuove** (grafico a destra). Divido le query a caso in 50% train e 50% test. Costruisco T1 con i documenti più cliccati dalle query di **train** e misuro sulle query di **test**:

| \|T1\| / \|D\| | click di test coperti | query con ≥1 doc in T1 | query con **tutti** i doc in T1 |
|---|---|---|---|
| 1% | 35% | 42% | 28% |
| 5% | 64% | 70% | 54% |
| 10% | 77% | 83% | 69% |
| uniforme (riferimento) | = \|T1\|/\|D\| | | |

**Perché è utile per l'idea.**
- È la verifica diretta dell'ipotesi della sezione 6: una piccola frazione di D serve la maggior parte delle query.
- Funziona su query **non viste** in train, quindi T1 si può costruire da un campione.
- La colonna "tutti i doc in T1" corrisponde a "la query si ferma a T1" (recall 1 rispetto ai click).
- **È una stima per difetto.** ORCAS conta ogni query una volta sola. Nel traffico reale le query frequenti si ripetono molte volte e di solito puntano proprio ai documenti popolari, quindi lo skew reale è più forte.

**Limiti.** Un click non coincide con un nearest neighbor nello spazio vettoriale. Per confermare il risultato nello spazio degli embedding bisogna codificare i 3.2M documenti e le query. Il corpus non è quello dei vettori MSMARCO-WS che abbiamo già.

---

## 2. Frequenza dei neighbor esatti nei benchmark ANN (GT esistenti)

**Domanda.** Usando i GT esatti, ci sono vettori che compaiono come NN di molte query?
**Riferimento.** Se i neighbor fossero uniformi su N vettori, con k·nq posizioni quasi tutte sarebbero distinte.

![frequenza NN](results/fig_nn_frequency_all.png)

| GT | k | NN distinti | atteso se uniforme | copertura test con NN di train (% di D usato) | lift |
|---|---|---|---|---|---|
| DEEP10M (10k q) | 10 | 98.9% | 99.5% | 1.0% (0.50%) | 2.1× |
| DEEP100M (10k q) | 100 | 99.0% | 99.5% | 1.0% (0.50%) | 2.1× |
| TTI10M (10k q) | 100 | 87.9% | 95.2% | 12.4% (4.65%) | 2.7× |
| TTI100M (10k q) | 100 | 97.0% | 99.5% | 3.1% (0.49%) | 6.3× |
| **TTI100M (100k q)** | 100 | **68.7%** | 95.2% | **34.5% (4.0%)** | **8.6×** |
| MSMARCO-WS 10M (9.4k q) | 10 | 97.2% | 99.5% | 2.7% (0.46%) | 5.8× |
| MSMARCO-WS 100M (9.4k q) | 10 | 99.3% | 99.95% | 0.9% (0.05%) | 18× |

"Lift" è la copertura ottenuta divisa per la frazione di D usata: un valore di 1 corrisponde all'uniforme.

**Cosa mostra.**
- **DEEP:** le query provengono dalla stessa distribuzione della base, e i neighbor sono praticamente uniformi. È il caso che la proposta descrive come "assunzione implicita dei benchmark".
- **TTI e MSMARCO:** le query sono fuori distribuzione (testo contro immagini, query contro documenti) e compare **hubness**: alcuni vettori sono vicini a molte query. Questo succede **anche senza query ripetute** ed è una proprietà della geometria, quindi aiuta T1 indipendentemente dal traffico.
- **L'effetto cresce con il numero di query.** Su TTI100M, passando da 10k a 100k query, T1 cresce di 8× (da 0.49% a 4.0% di D), ma la copertura cresce di 11× (da 3.1% a 34.5%). Inoltre il 35.5% delle query di test ha almeno metà dei 100-NN in T1. Con un log reale da milioni di query l'effetto dovrebbe essere ancora maggiore.
- **Le query di MSMARCO-WS sono deduplicate:** 9368 vettori distinti su 9376, e la similarità coseno mediana con la query più vicina è 0.80. Su questo benchmark la frequenza delle query non è misurabile.

**Perché è utile per l'idea.**
- Spiega perché i benchmark **non sono adatti** a valutare il tiering: lo skew è stato rimosso dal campionamento. Serve quindi un workload derivato da log reali (sezione 4).
- Mostra che sulle query fuori distribuzione, che sono il caso realistico per text retrieval e cross-modal, c'è comunque una concentrazione sfruttabile.

---

## 3. Località spaziale: i NN di una query stanno nella stessa zona?

**Setup.** Uso MSMARCO-WS 10M (768 dimensioni) e un k-means L2 con C = 1024 e C = 4096 celle, addestrato su un campione della base. Assegno tutti i 10M vettori e conto in quali celle cadono i 10-NN esatti delle 9376 query.

![località spaziale](results/fig_msmarco_spatial.png)

| metrica (C = 1024) | valore |
|---|---|
| celle distinte occupate dai 10-NN di una query | **5.7** in media (mediana 6) |
| stessa cosa per 10 documenti a caso | 9.9 |
| massimo numero di NN della query nella stessa cella | 4.3 su 10 |
| query con tutti i 10-NN in ≤2 celle | 10% |
| NN che cadono nella cella più vicina alla query | 26% |
| celle che contengono almeno un NN | 927 su 1024 |

**T1 costruito per celle.** Ordino le celle per numero di NN delle query di train, misuro sulle query di test:

| \|T1\| / \|D\| | NN di test coperti | query con recall@10 = 1 solo su T1 | query con recall@10 ≥ 0.9 |
|---|---|---|---|
| 1% | 4.8% | 0.5% | 1.0% |
| 5% | 16.7% | 2.5% | 4.9% |
| 10% | 25.5% | 4.6% | 8.9% |
| 20% | 38.9% | 8.8% | 15.6% |
| 30% | 49.5% | 12.8% | 22.0% |

Con C = 4096 i risultati sono quasi identici.

**Cosa mostra.**
- I NN **sono localizzati**: occupano circa metà delle celle che occuperebbero punti a caso (grafico centrale). La curva di Lorenz (grafico a destra) mostra che la massa dei NN è concentrata in una parte dello spazio: il 10% di D ne contiene il 25%, il 30% di D ne contiene il 50%.
- Costruire T1 **per cluster** generalizza a query nuove molto meglio che per singolo vettore. Con meno dello 0.5% di D, per singolo vettore si copre il 2.7%; per cluster, con l'1% si copre il 4.8% e con il 10% il 25%.
- Però, **con query campionate in modo uniforme**, quasi nessuna query si può chiudere su T1: solo il 4.6% con il 10% di D. La località da sola non basta.

**Perché è utile per l'idea.**
- Giustifica la costruzione dei livelli a granularità di **cluster o partizione** invece che di singolo vettore. È più economico, ha più senso come indice (ogni livello è un IVF o HNSW su un sottoinsieme coerente) e generalizza meglio.
- Fornisce feature naturali per lo stimatore di recall: la distanza della query dai centroidi di T1, o la presenza dei NN nella cella della query.
- Insieme alla sezione 2, mostra che **il beneficio grande deve venire dalla frequenza del traffico**, che c'è nei log (sezione 1) ma non nei benchmark.

---

## 4. Messaggio per il meeting

1. **L'ipotesi è supportata dove conta, cioè nel traffico reale.** ORCAS mostra che il 10% del corpus serve il 77% dei click e chiude il 69% delle query nuove, e questo è una stima per difetto.
2. **I benchmark ANN standard eliminano lo skew** (query campionate e deduplicate). È un argomento a favore del nostro posizionamento ("i benchmark assumono query uniformi"), ma significa anche che **dobbiamo costruire un workload da log reali**.
3. **La geometria aiuta, ma non basta.** Hubness e località spaziale danno un lift da 2× a 9× rispetto all'uniforme, ma senza skew di frequenza poche query si chiudono su T1.

### Osservazione sulla formulazione (sezione 4 della proposta)
L'obiettivo Σ_q w(q)·|S ∩ NN_k(q)|/k è **lineare (modulare)** in S, non solo submodulare: equivale a Σ_{x∈S} Σ_{q: x∈NN(q)} w(q)/k. L'ottimo esatto è quindi semplicemente prendere i B vettori con frequenza pesata più alta, senza bisogno di greedy né del fattore (1−1/e).
I dati però mostrano che la metrica che conta per fermarsi a T1 è "**tutti** i NN della query in T1", e questa è molto più severa: su ORCAS al 10% passa dal 77% (click coperti) al 69% (query chiuse); su MSMARCO-WS dal 25.5% al 4.6%.
Un obiettivo che premi le query **completamente coperte** è un problema di tipo budgeted max-coverage su insiemi (una query conta solo se tutto NN(q) è in S). Lì la struttura combinatoria diventa interessante, ed è un possibile contributo.

### Prossimi passi proposti
1. **Embedding di ORCAS** (3.2M documenti più un campione di query) con un encoder denso, per ripetere i test 2 e 3 nello spazio vettoriale con traffico reale: è la prova che unisce le due cose.
2. **TripClick** (log grezzo con query ripetute e timestamp) per misurare la frequenza delle query (curva rango–frequenza, H(p)) e il **drift** nel tempo. Va richiesto subito.
3. Workload sintetico sui benchmark: query campionate con distribuzione Zipf sulle query esistenti, per avere subito esperimenti su latenza e recall con gli indici che abbiamo già.

---
*Riproducibilità.* Gli script sono in `exp_eval/skew/`:
- `orcas_skew.py`: test 1
- `nn_freq_all.py`: test 2
- `msmarco_skew.py` e `run_msmarco_skew.slurm`: test 3, job 6997502, circa 5 minuti
- `plot_skew.py`: grafici

I risultati JSON e i grafici PDF sono in `exp_eval/skew/results/`. Lo split train/test è 50/50 casuale con seed 0, su un solo seed.
