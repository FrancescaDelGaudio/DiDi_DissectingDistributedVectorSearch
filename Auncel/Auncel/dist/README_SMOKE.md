# Auncel dist — smoke test (toy dataset)

Dataset di default:

- `/gpfs/scratch/delgaudio/vectors/toy/generated_deep_files/random_base_100.fbin`
- `random_queries_10_k5.fbin`
- `random_GT_10_k5.bin`

Parametri di default nel codice:

| Parametro | Valore |
|---|---|
| Indice | `IVF8,Flat` |
| k | 5 (dal GT) |
| nq | 10 |
| required_recall | 0.9 (error bound 10%) |
| hyperparam_id | 1 |
| porta worker | 3456 |

## 1. Converti i dati

```bash
python3 dist/scripts/convert_toy_data.py
```

Crea in `.../pyramid/`: `base.fvecs`, `query.fvecs`, `idx.ivecs`, `dis.fvecs`.

## 2. Compila

```bash
module load intel/20U4/mkl
export LD_LIBRARY_PATH=$MKLROOT/lib/intel64:$LD_LIBRARY_PATH

cd /gpfs/users/delgaudio/exp_eval/Auncel/Auncel
./configure --without-cuda
make -j4
make install
cd dist && make -j4
```

Binari: `dist/worker`, `dist/master`, `dist/gt`, `dist/split`, `dist/reduce`.

## 3. Smoke test — 1 worker

```bash
export LD_LIBRARY_PATH=$MKLROOT/lib/intel64:$LD_LIBRARY_PATH
cd dist

# Terminal 1
./worker

# Terminal 2
./master train 127.0.0.1:3456
./master search ./results 127.0.0.1:3456
```

Il worker stampa recall@k per query dopo `search`.

## 4. Smoke test — 2 worker (opzionale)

```bash
python3 scripts/convert_toy_data.py  # base globale in pyramid/

./split ../vectors/toy/.../random_base_100.fbin ./shards/ 0
# oppure path assoluto al fbin

./gt ./shards/0db.fbin
./gt ./shards/1db.fbin

# Terminal 1 — shard 0
./worker 3456 ./shards/0db.fvecs ...  # converti shard in fvecs prima

# Per shard usare gt locale:
#   gtI=./shards/0gti.ivecs  gtD=./shards/0gtd.fvecs
```

Per 2 worker converti ogni `Xdb.fbin` in `Xdb.fvecs` e lancia worker con GT locale.

```bash
./master train 127.0.0.1:3456 127.0.0.1:3457
./master search ./results 127.0.0.1:3456 127.0.0.1:3457
./reduce ./results 2 0 50
```

## CLI worker

```
./worker [port] [db.fvecs] [query.fvecs] [gtI.ivecs] [gtD.fvecs] \
         [index_key] [required_recall] [hyperparam_id] [omp_threads]
```

Esempio shard 1:

```bash
./worker 3457 \
  /path/to/shards/1db.fvecs \
  /path/to/pyramid/query.fvecs \
  /path/to/shards/1gti.ivecs \
  /path/to/shards/1gtd.fvecs
```
