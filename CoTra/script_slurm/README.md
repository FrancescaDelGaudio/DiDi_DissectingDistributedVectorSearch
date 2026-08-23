# CoTra on Slurm (stella / ncpum)

CoTra was designed for fixed host lists. On Slurm, **nodes are chosen at runtime**.
These scripts generate the `.conf` file from `$SLURM_JOB_NODELIST` inside each job.

## Prerequisites

1. `MACHINE_NUM` in `include/rdma/rdma_config.h` must equal `#SBATCH -N` (default **4**).
2. Rebuild CoTra after changing `MACHINE_NUM`.
3. User-space deps (memcached, libmemcached, libaio headers, gperftools, fmt):
   `bash script_slurm/build_cotra_deps.sh` → `/gpfs/scratch/delgaudio/cotra/deps`
4. Faiss (CPU, gcc 10): run once
   `/gpfs/users/delgaudio/exp_eval/Distributed-HNSW/script/build_faiss_gcc10.sh`

## Build CoTra (login node)

```bash
cd /gpfs/users/delgaudio/exp_eval/CoTra
bash script_slurm/build_cotra.sh
# binaries: build/tests/scala_index, build/tests/scala_anns
```

CMake paths are in `cmake/StellaCluster.cmake` (MKL under `/gpfs/softs/intel/oneapi`, RDMA under `/usr/lib64`, etc.).
Override with `-DVAR=...` if needed.

## Workflow

```bash
cd /gpfs/users/delgaudio/exp_eval/CoTra
mkdir -p script_slurm/logs script_slurm/runtime

# 1) Probe RDMA
module purge && module load gcc/10.3.0
sbatch script_slurm/check_rdma.slurm

# 2) Test conf generation + memcached (no index build yet)
sbatch script_slurm/setup_cluster.slurm   # uses ncpumshort (quick preflight)
cat script_slurm/logs/setup_cluster-<JOBID>.out

# 3) Build index (DEEP 10M, 4 nodes)
sbatch script_slurm/scala_index_deep10M.slurm
tail -f script_slurm/logs/scala_index_deep10M-<JOBID>.out

# 4) Query (after index): same -N 4, same index_path_prefix, new conf per job
```

## 6 partitions (separate build tree)

```bash
bash script_slurm/build_cotra_6p.sh
sbatch script_slurm/scala_index_deep10M_6p.slurm   # -N 6, output .../deep10M_6p/
sbatch script_slurm/scala_index_tti10M_6p.slurm
bash script_slurm/submit_anns_sweep_6p.sh
```

Binaries: `build-6p/tests/scala_index_6p` and `scala_anns_6p` (4p jobs keep using `build/tests/scala_index`).

## All four index jobs (disjoint nodes)

CoTra RDMA jobs **must not share compute nodes** (RDMA `PART_INFO` / memcached cross-talk).

Each index Slurm script now includes `#SBATCH --exclusive`: Slurm allocates whole nodes and will not place another job on them.

```bash
bash script_slurm/build_cotra.sh
bash script_slurm/build_cotra_6p.sh
bash script_slurm/submit_all_index_jobs.sh          # parallel: needs 20 nodes (4+4+6+6)
# or
bash script_slurm/submit_all_index_jobs.sh --sequential   # one at a time, peak 6 nodes
```

After jobs are **RUNNING**, verify no node overlap:

```bash
bash script_slurm/check_index_job_nodes.sh
```

If the cluster has fewer than 20 free nodes, parallel jobs queue until enough exclusive nodes are available (they will not share nodes when running together).

## Files

| File | Role |
|------|------|
| `generate_cluster_conf.sh` | Writes `runtime/cluster-job<JOBID>.conf` from Slurm nodelist |
| `start_memcached_leader.sh` | Starts memcached on leader (sorted first hostname) |
| `env_slurm.sh` | `module load` shared by jobs |
| `check_rdma.slurm` | IB probe on all allocated nodes |
| `setup_cluster.slurm` | End-to-end test of conf + memcached |
| `scala_index_deep10M.slurm` | Distributed `scala_index` on DEEP `base.10M.fbin` (4 partitions) |
| `build_cotra_6p.sh` | Build `scala_index_6p` / `scala_anns_6p` in `build-6p/` |
| `scala_index_deep10M_6p.slurm` | DEEP L2 indexing, 6 nodes |
| `scala_index_tti10M_6p.slurm` | TTI MIPS indexing, 6 nodes |
| `scala_anns_6p.slurm` | Query sweep helper for 6 partitions |
| `submit_all_index_jobs.sh` | Submit DEEP/TTI × 4p/6p with `--exclusive` |
| `check_index_job_nodes.sh` | Assert no shared nodes between cotra_idx jobs |

## Conf format (generated)

```
<leader_hostname>    # machine 0, memcached binds here
2378                 # memcached port
<node0>=0
<node1>=1
...
```

Leader = **lexicographically first** hostname in the allocation (`sort`).
Machine IDs are stable within the job.

## Important

- Submit **the same `-N`** every time CoTra runs; it must match `MACHINE_NUM`.
- Use the **same `SLURM_JOB_ID` conf** only within one job. Each new job regenerates conf.
- All tasks must use the same `--config_file $CONF_FILE` via `srun`.
- **IB device on ncpum is `mlx5_0`** (not `mlx4_0`). Slurm scripts pass `-d $COTRA_IB_DEV` (default `mlx5_0` in `env_slurm.sh`). After changing RDMA code, rebuild with `bash script_slurm/build_cotra.sh`.
- **Never run two CoTra index jobs on overlapping nodes.** Use `#SBATCH --exclusive` (already in the index scripts) or `submit_all_index_jobs.sh`.
