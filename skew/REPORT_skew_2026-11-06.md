# Preliminary study: are queries and neighbors concentrated?

November 6, 2026. This is Section 6 of the secondment proposal.

**Summary**
1. **Real logs are strongly skewed.** On ORCAS (real Bing clicks), a first tier T1 holding **10% of the documents** covers **77% of the clicks** of unseen queries, and fully answers **69% of the queries**. A uniform distribution would give 10%.
2. **ANN benchmarks have no query skew, by construction.** Their queries are sampled and deduplicated. Exact neighbors repeat little there: almost not at all on DEEP, moderately on TTI and MSMARCO (out-of-distribution queries).
3. **Neighbors are clustered in space.** On MSMARCO-WS 10M, the 10 NN of a query lie on average in **5.7 out of 1024 k-means cells**, versus 9.9 for 10 random documents. The 10% of the space chosen from training queries holds **25%** of the test queries' NN. That is still too little to answer queries from T1 alone when traffic is uniform.

**What this means for the idea:** the gain has to come from the skew of real traffic, which benchmarks remove. Spatial locality is an extra advantage: it lets us build tiers per cluster and generalize to new queries.

---

## 1. ORCAS: document frequency in real clicks

**Data.** ORCAS contains 18.8M (query, clicked document) pairs from Bing logs, on the MS MARCO Document corpus (3.21M documents). There are 10.4M queries, all distinct, and no timestamps, users or click counts.
I use the clicked document as the "relevant neighbor" of the query.

![ORCAS](results/fig_orcas.png)

**Document popularity** (left plot, log-log axes):
- Only 44% of the corpus (1.42M documents) receives at least one click; 56% is never clicked.
- The rank–frequency curve is close to a power law. The most popular document (facebook.com) is clicked from 11,310 distinct queries; 355k documents from a single one.
- Share of query–document pairs covered by the most popular documents:

  | top % of corpus | 0.1% | 1% | 5% | 10% | 20% |
  |---|---|---|---|---|---|
  | share of clicks | 12% | 36% | 65% | 79% | 92% |

  The Gini index over the whole corpus is 0.88.

**Generalization to new queries** (right plot). I split the queries at random into 50% train and 50% test. I build T1 from the documents most clicked by the **train** queries and measure on the **test** queries:

| \|T1\| / \|D\| | test clicks covered | queries with ≥1 doc in T1 | queries with **all** docs in T1 |
|---|---|---|---|
| 1% | 35% | 42% | 28% |
| 5% | 64% | 70% | 54% |
| 10% | 77% | 83% | 69% |
| uniform (reference) | = \|T1\|/\|D\| | | |

**Why this matters for the idea.**
- It is the direct check of the Section 6 hypothesis: a small fraction of D serves most of the queries.
- It works on queries **not seen** in training, so T1 can be built from a sample.
- The column "all docs in T1" corresponds to "the query stops at T1" (recall 1 with respect to the clicks).
- **It is a lower bound.** ORCAS counts each query only once. In real traffic frequent queries repeat many times and usually point to the popular documents, so the real skew is stronger.

**Limitations.** A click is not a nearest neighbor in vector space. Confirming the result in embedding space requires encoding the 3.2M documents and the queries. The corpus is not the one behind the MSMARCO-WS vectors we already have.

---

## 2. Exact-neighbor frequency in ANN benchmarks (existing GTs)

**Question.** Using the exact GTs, are there vectors that appear as NN of many queries?
**Reference.** If neighbors were uniform over N vectors, almost all of the k·nq slots would be distinct.

![NN frequency](results/fig_nn_frequency_all.png)

| GT | k | distinct NN | expected if uniform | test coverage by train NN (% of D used) | lift |
|---|---|---|---|---|---|
| DEEP10M (10k q) | 10 | 98.9% | 99.5% | 1.0% (0.50%) | 2.1× |
| DEEP100M (10k q) | 100 | 99.0% | 99.5% | 1.0% (0.50%) | 2.1× |
| TTI10M (10k q) | 100 | 87.9% | 95.2% | 12.4% (4.65%) | 2.7× |
| TTI100M (10k q) | 100 | 97.0% | 99.5% | 3.1% (0.49%) | 6.3× |
| **TTI100M (100k q)** | 100 | **68.7%** | 95.2% | **34.5% (4.0%)** | **8.6×** |
| MSMARCO-WS 10M (9.4k q) | 10 | 97.2% | 99.5% | 2.7% (0.46%) | 5.8× |
| MSMARCO-WS 100M (9.4k q) | 10 | 99.3% | 99.95% | 0.9% (0.05%) | 18× |

"Lift" is the coverage obtained divided by the fraction of D used: a value of 1 means uniform.

**What it shows.**
- **DEEP:** queries come from the same distribution as the base, and neighbors are practically uniform. This is the case the proposal describes as the "implicit assumption of benchmarks".
- **TTI and MSMARCO:** queries are out of distribution (text vs images, queries vs documents) and **hubness** appears: some vectors are close to many queries. This happens **even without repeated queries** and is a property of the geometry, so it helps T1 regardless of traffic.
- **The effect grows with the number of queries.** On TTI100M, going from 10k to 100k queries, T1 grows 8× (from 0.49% to 4.0% of D), but coverage grows 11× (from 3.1% to 34.5%). Moreover, 35.5% of test queries have at least half of their 100-NN in T1. With a real log of millions of queries the effect should be even larger.
- **MSMARCO-WS queries are deduplicated:** 9368 distinct vectors out of 9376, and the median cosine similarity with the nearest other query is 0.80. Query frequency cannot be measured on this benchmark.

**Why this matters for the idea.**
- It explains why benchmarks are **not suited** to evaluate tiering: the skew has been removed by sampling. We therefore need a workload derived from real logs (Section 4).
- It shows that on out-of-distribution queries, the realistic case for text and cross-modal retrieval, there is still an exploitable concentration.

---

## 3. Spatial locality: do a query's NN lie in the same region?

**Setup.** I use MSMARCO-WS 10M (768 dimensions) and an L2 k-means with C = 1024 and C = 4096 cells, trained on a sample of the base. I assign all 10M vectors and count which cells contain the exact 10-NN of the 9376 queries.

![spatial locality](results/fig_msmarco_spatial.png)

| metric (C = 1024) | value |
|---|---|
| distinct cells occupied by a query's 10-NN | **5.7** on average (median 6) |
| same for 10 random documents | 9.9 |
| maximum number of the query's NN in a single cell | 4.3 out of 10 |
| queries with all 10-NN in ≤2 cells | 10% |
| NN falling in the cell nearest to the query | 26% |
| cells containing at least one NN | 927 of 1024 |

**T1 built from cells.** I rank cells by the number of train-query NN they hold and measure on test queries:

| \|T1\| / \|D\| | test NN covered | queries with recall@10 = 1 on T1 alone | queries with recall@10 ≥ 0.9 |
|---|---|---|---|
| 1% | 4.8% | 0.5% | 1.0% |
| 5% | 16.7% | 2.5% | 4.9% |
| 10% | 25.5% | 4.6% | 8.9% |
| 20% | 38.9% | 8.8% | 15.6% |
| 30% | 49.5% | 12.8% | 22.0% |

With C = 4096 the results are almost identical.

**What it shows.**
- NN **are localized**: they occupy about half as many cells as random points would (middle plot). The Lorenz curve (right plot) shows that the NN mass is concentrated in part of the space: 10% of D holds 25% of it, 30% of D holds 50%.
- Building T1 **per cluster** generalizes to new queries much better than per single vector. With less than 0.5% of D, the per-vector T1 covers 2.7%; per cluster, 1% covers 4.8% and 10% covers 25%.
- However, **with uniformly sampled queries**, almost no query can be closed on T1: only 4.6% with 10% of D. Locality alone is not enough.

**Why this matters for the idea.**
- It justifies building the tiers at **cluster or partition** granularity rather than per vector. This is cheaper, makes more sense as an index (each tier is an IVF or HNSW over a coherent subset) and generalizes better.
- It provides natural features for the recall estimator: the query's distance to the T1 centroids, or whether the NN fall in the query's cell.
- Together with Section 2, it shows that **the big gain must come from traffic frequency**, which is present in logs (Section 1) but not in benchmarks.

---

## 4. Message for the meeting

1. **The hypothesis is supported where it matters, i.e. in real traffic.** ORCAS shows that 10% of the corpus serves 77% of the clicks and closes 69% of new queries, and this is a lower bound.
2. **Standard ANN benchmarks remove the skew** (sampled and deduplicated queries). This supports our positioning ("benchmarks assume uniform queries"), but it also means **we must build a workload from real logs**.
3. **Geometry helps, but is not enough.** Hubness and spatial locality give a 2× to 9× lift over uniform, but without frequency skew few queries close on T1.

### A remark on the formulation (Section 4 of the proposal)
The objective Σ_q w(q)·|S ∩ NN_k(q)|/k is **linear (modular)** in S, not just submodular: it equals Σ_{x∈S} Σ_{q: x∈NN(q)} w(q)/k. The exact optimum is therefore simply to take the B vectors with the highest weighted frequency; neither greedy nor the (1−1/e) factor is needed.
The data show, however, that the metric that matters for stopping at T1 is "**all** of the query's NN in T1", and it is much harsher: on ORCAS at 10% it drops from 77% (clicks covered) to 69% (queries closed); on MSMARCO-WS from 25.5% to 4.6%.
An objective that rewards **fully covered** queries is a budgeted max-coverage problem over sets (a query counts only if all of NN(q) is in S). That is where the combinatorial structure becomes interesting, and it is a possible contribution.

### Proposed next steps
1. **Embed ORCAS** (3.2M documents plus a sample of queries) with a dense encoder, to repeat tests 2 and 3 in vector space with real traffic: this is the experiment that brings the two together.
2. **TripClick** (raw log with repeated queries and timestamps) to measure query frequency (rank–frequency curve, H(p)) and **drift** over time. Access should be requested now.
3. Synthetic workload on the benchmarks: queries sampled with a Zipf distribution over the existing queries, so we can immediately run latency and recall experiments with the indexes we already have.

---
*Reproducibility.* Scripts are in `exp_eval/skew/`:
- `orcas_skew.py`: test 1
- `nn_freq_all.py`: test 2
- `msmarco_skew.py` and `run_msmarco_skew.slurm`: test 3, job 6997502, about 5 minutes
- `plot_skew.py`: plots

JSON results and PDF plots are in `exp_eval/skew/results/`. The train/test split is a random 50/50 with seed 0, on a single seed.
