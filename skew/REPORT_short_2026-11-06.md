# Is there skew to exploit? Preliminary results

**Question:** does a small fast tier T1 hold the neighbors of most queries?
**Method:** split queries 50/50 into train and test, build T1 from the train queries, measure on the unseen test queries. Uniform = no skew.

## Datasets and experiments

| Dataset | What it is | Experiment |
|---|---|---|
| **ORCAS** | 18.8M real Bing (query, clicked doc) pairs, 3.2M-doc corpus | Doc popularity; T1 = most-clicked docs |
| **DEEP, TTI, MSMARCO-WS** (10M/100M) | Standard ANN benchmarks, exact top-10/100 GT | How often the same NN serve different queries |
| **MSMARCO-WS 10M** | 768-d text embeddings, 9.4k queries | k-means (1024 cells): are a query's NN in the same region? T1 = top cells |

## 1. ORCAS (real clicks): works

![](results/s_orcas.png)

A T1 with **10% of the docs** covers **77%** of the clicks and fully answers **69%** of new queries. This is a lower bound, because ORCAS counts each query only once.

## 2. ANN benchmark GTs: does not work as is

![](results/s_benchmarks.png)

Benchmark queries are sampled and deduplicated, so the true NN almost never repeat across queries (<3%). Reuse grows with traffic volume: on TTI with 100k instead of 10k queries, coverage goes from 3% to 35%.

## 3. Spatial locality (MSMARCO-WS 10M): helps, not enough alone

![](results/s_msmarco_spatial.png)

A query's 10-NN fall in 5.7 cells on average, against 9.9 for random docs, so it makes sense to build tiers by cluster. Without frequency skew, however, T1 fully answers only about 5% of queries.

## For the idea

- **Real traffic is strongly skewed (ORCAS).** This is the core assumption, and it holds on held-out queries.
- **Neighbors are spatially grouped.** Tiers can be built per cluster, and clusters give features for the recall estimator.
- **Benchmarks remove the skew.** This supports our positioning ("ANN work assumes uniform queries").

## Problems / open points

- **ORCAS has no vectors.** We must embed its 3.2M docs and the queries with a dense encoder; this needs GPU time. Clicks are also only a proxy for NN.
- **ORCAS has no timestamps, users or query counts**, so we cannot measure query frequency or drift from it. **TripClick** has both; access must be requested.
- **Benchmarks cannot show the effect.** We need a real-log workload, or Zipf-sampled queries over the benchmark queries.
- **The tier must be judged on "all NN in T1".** That is much stricter than "NN covered" (69% vs 77% on ORCAS, 5% vs 25% on MSMARCO).
- **The T1 objective in the proposal is linear, not just submodular.** Taking the top-B by weighted frequency is already optimal. A "fully answered queries" objective is the harder and more interesting one.
