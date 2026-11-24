# CSCI 423/523 · CUDA C++ midterm prep

Solutions to past midterms, 10 practice midterms with solutions and an auto-grader, standalone code templates, and a theory guide (in Russian) for the Parallel Systems and GPU Programming course.

These are our own solutions, not official ones. Everything compiles against CUDA 12 headers (the CUDA 13 code path was checked against CUDA 13.5 headers) and was run end to end in a CPU emulation of CUDA (HIP-CPU), where every program prints `PASSED`, both with and without managed-memory prefetch support. Run them on a real GPU before you rely on them.

## Layout

| Path | What it is |
| --- | --- |
| `STARTER.cu` | Universal starter for any task: everything that is the same in every solution, with `CHANGE` comments marking the task-specific lines. Compiles and runs as is |
| `THEORY.md` | Theory from zero: host/device, thread hierarchy, grid-stride loops, memory, streams, timing, reduction, matrices, common bugs, API cheat sheet, checklist |
| `midterms/2025-midterm-1-matmul/` | 2025 midterm 1: N×N integer matrix multiplication with streams |
| `midterms/2026-midterm-1-vectors/` | 2026 midterm 1: add / subtract / multiply two arrays + reduction |
| `midterms/2026-midterm-2-montecarlo-matmul/` | 2026 midterm 2 (originally Python/Numba) solved in CUDA C++: Monte Carlo π with cuRAND, random matrix multiplication + reduction, largest N |
| `practice/TASKS.md` | 10 practice midterms in the same format |
| `practice/solutions/` | One solution file per practice task |
| `practice/trainer.py` | Terminal grader: compiles, runs and grades a solution out of 20 |
| `practice/trainer.html` | Same tasks in a browser terminal with vim keys (static grading only) |
| `templates/` | Standalone building blocks, each one a small runnable program |

## Build and run

```bash
nvcc -O2 -o prog path/to/file.cu
./prog
compute-sanitizer ./prog      # out-of-bounds and race errors
```

The 2026 midterm 2 solution uses cuRAND's device API, which is header-only: no extra link flags needed.

Every file that uses managed memory goes through a small `prefetch()` helper instead of calling `cudaMemPrefetchAsync` directly, for two reasons that matter in the exam lab (WSL, Ubuntu 26.04):

- CUDA 13 changed the signature of `cudaMemPrefetchAsync` (it now takes a `cudaMemLocation`), so the old four-argument call does not compile there. The helper picks the right form with `CUDART_VERSION`.
- WSL has only limited managed-memory support (`cudaDevAttrConcurrentManagedAccess` is 0), where `cudaMemPrefetchAsync` returns an error. The helper skips the prefetch in that case instead of aborting. Always synchronize before the CPU touches managed memory: on WSL, CPU access while a kernel is running crashes the program.

## Practice

```bash
cd practice
python3 trainer.py list                    # tasks and your best scores
python3 trainer.py show 3                  # print task 3
python3 trainer.py exam                    # random task + 90-minute timer
python3 trainer.py check 3 yourname.cu     # compile, run, grade /20
python3 trainer.py check 3 yourname.cu --static   # rubric only, no GPU needed
```

Every task fills its arrays with fixed formulas, so the answers are known in advance:

| # | Task | Checked values |
| --- | --- | --- |
| 1 | Vector add / sub / mul | Sum (addition) = -6, Sum (subtraction) = -12, Sum (multiplication) = -4194310 |
| 2 | Matrix multiplication | Sum (C) = -12582144, C[0][0] = -504, C[N-1][N-1] = 126 |
| 3 | Integer SAXPY in place | Sum (y) = -50, y[12345] = 7 |
| 4 | Dot product + max/min | Dot = -32, Max (a*b) = 32, Min (a*b) = -32 |
| 5 | C = A + Bᵀ | Sum (C) = -524292, Trace (C) = -2053 |
| 6 | Polynomial + atomic count | Sum (y) = -122689036, Count (y > 0) = 399456 |
| 7 | Shared-memory histogram | Bin -5..5 ≈ 381300 each, Sum = -4 |
| 8 | Sum / min / max | Sum = -30759, Min = -500, Max = 500 |
| 9 | 1D stencil | Sum (c) = -15, c[0] = -11, c[N-1] = 8, c[1000] = -12 |
| 10 | Matrix-vector | Sum (y) = 9, Max (y) = 14, y[7] = 11 |

Try a task yourself before opening its solution.

## Templates

| File | Shows |
| --- | --- |
| `01_skeleton_vector_add.cu` | Full program covering every grading point: SM-based blocks, streams, prefetch, grid-stride, device function, reduction, timing, verify, cleanup |
| `02_reduce_sum.cu` | Shared-memory tree reduction + `atomicAdd` |
| `03_reduce_max_min.cu` | Max / min reduction with `atomicMax` / `atomicMin` |
| `04_reduce_two_pass.cu` | Reduction without atomics: per-block partial sums |
| `05_histogram_atomics.cu` | Privatized histogram in shared memory, atomic counter |
| `06_memcpy_async_streams.cu` | `cudaMalloc` + pinned host memory + `cudaMemcpyAsync` |
| `07_matmul_2d_grid.cu` | 2D blocks and a 2D grid-stride loop |
| `08_largest_n.cu` | Largest N×N problem that fits in GPU memory |
