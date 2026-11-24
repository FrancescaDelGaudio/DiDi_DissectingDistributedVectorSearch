# Is there skew to exploit? Preliminary results

**Question:** does a small fast tier T1 hold the neighbors of most queries?
**Method:** split queries into train and test, build T1 from the train queries, measure on the unseen test queries. Uniform = no skew.
**Key metric:** % of queries *fully* answered by T1 (recall@10 ≥ 0.9 or = 1 on T1 alone), i.e. queries that do not need to descend. It is the orange line in the plots.

## Datasets and experiments

| Dataset | What it is | Experiment |
|---|---|---|
| **ORCAS** | 18.8M real Bing (query, clicked doc) pairs, 3.2M-doc corpus | Doc popularity; T1 = most-clicked docs |
| **DEEP, TTI, MSMARCO-WS** (10M/100M) | Standard ANN benchmarks, exact top-10/100 GT | How often the same NN serve different queries |
| **MSMARCO-WS 10M** | 768-d text embeddings, 9.4k queries | k-means (1024 cells): are a query's NN in the same region? T1 = top cells |
| **BEIR** (FEVER, Climate-FEVER, Quora) | Text retrieval benchmarks; precomputed embeddings (Cohere, 1024-d); we compute the exact top-10 GT | Topical locality, simulated drift, tier vs semantic cache, effect of traffic volume |

## 1. ORCAS (real clicks): works

![](results/s_orcas.png)

A T1 with **10% of the docs** covers **77%** of the clicks and fully answers **69%** of new queries. This is a lower bound, because ORCAS counts each query only once.

## 2. ANN benchmark GTs: does not work as is

![](results/s_benchmarks.png)

Benchmark queries are sampled and deduplicated, so the true NN almost never repeat across queries (<3%). Reuse grows with traffic volume: on TTI with 100k instead of 10k queries, coverage goes from 3% to 35%.

## 3. Spatial locality (MSMARCO-WS 10M): helps, not enough alone

![](results/s_msmarco_spatial.png)

A query's 10-NN fall in 5.7 cells on average, against 9.9 for random docs, so it makes sense to build tiers by cluster. Without frequency skew, however, T1 fully answers only about 5% of queries.

## 4. BEIR: controlled experiments on context, drift and caching

**Setup.**
- Corpus: 5.4M Wikipedia pages, shared by two query sets.
  - **FEVER**: generic claims on any topic; 110k train and 6.7k test queries.
  - **Climate-FEVER**: 1.5k claims, all about climate change.
- **Quora**: 523k questions; each test question has duplicates (same intent, other wording) in the corpus.
- Embeddings from Cohere embed-english-v3; we computed the exact top-10 GT.

![](results/s_beir_e1.png)

- **Topical locality works.** For the climate queries, a T1 with **5% of the corpus** serves **82%** of the queries; for the generic FEVER queries the same budget serves only **15%**. 80% of the climate NN lie in 1.6% of the corpus. A T1 built from the FEVER queries serves 0% of the climate queries.
- **Drift detection and re-layering work (simulated).** The query stream switches from FEVER to Climate-FEVER at window 30, and ρ (the share of queries descending below T1) jumps from 83% to 100%. Rebuilding T1 from the last 3 windows brings it down to 47%.

![](results/s_beir_e2e3.png)

- **The tier beats the semantic cache.** Here the "earlier query" is a Quora duplicate of the test query. The two wordings share only 64% of their top-10. At equal memory:
  - the cache serves at most 20% of the queries, and at threshold 0.90 many of its hits are wrong;
  - a T1 holding the **same documents**, searched with the new query, serves **36%**.
- **More traffic means more reuse.** On FEVER, going from 1k to 110k train queries raises the test NN covered by T1 from 6% to 40%, and the curve is not saturated.
- **Does not work.** Generic FEVER queries reach only 10% fully answered even with 110k queries. The BEIR dev/test splits contain no repeated queries. On Climate-FEVER the encoder places only 27% of the relevant docs in the top-10 (hard task, so NN ≠ relevant there).

## For the idea

- **Real traffic is strongly skewed (ORCAS).** This is the core assumption, and it holds on held-out queries.
- **Neighbors are spatially grouped.** Tiers can be built per cluster, and clusters give features for the recall estimator.
- **Context makes the skew large (BEIR).** Queries from one topic or user group live in a small region, so per-context tiers are very effective.
- **Re-layering via ρ works,** and **a tier beats a semantic cache** at equal memory (BEIR).
- **Benchmarks remove the skew.** This supports our positioning ("ANN work assumes uniform queries").

## Problems / open points

- **ORCAS has no vectors.** We must embed its 3.2M docs and the queries with a dense encoder; this needs GPU time. Clicks are also only a proxy for NN.
- **ORCAS has no timestamps, users or query counts**, so we cannot measure query frequency or drift from it. **TripClick** has both; access must be requested.
- **Benchmarks cannot show the effect.** We need a real-log workload, or Zipf-sampled queries over the benchmark queries.
- **The tier must be judged on "all NN in T1" (orange line).** That is much stricter than "NN covered" (69% vs 77% on ORCAS, 5% vs 25% on MSMARCO).
- **The BEIR drift is simulated by us** (one dataset after the other), and BEIR uses a different encoder (Cohere, 1024-d) than the one we will use for ORCAS and TripClick.

## Conclusion: what each dataset is for

| Dataset | Use it to show | Cannot show | Status |
|---|---|---|---|
| **ORCAS** | Real traffic concentrates on few documents (neighbor popularity); T1 generalizes to unseen queries | Query frequency, drift (no counts, no timestamps); vector-space NN until we embed it | Click analysis done; embeddings to do |
| **TripClick** | Real query frequency (repeated queries, head/torso/tail), real drift over 7 years (timestamps), sessions as context | Large scale (1.5M docs, medical domain only) | Access to request (Logs Dataset + IR Benchmark) |
| **BEIR** | Per-context tiers (topical locality), re-layering under drift, tier vs semantic cache, effect of traffic volume; ready-made embeddings, so it is our first testbed for the full multi-tier system | That skew exists in real traffic (no frequencies, no timestamps; drift simulated) | Done (FEVER, Climate-FEVER, Quora) |
