# CUDA midterm practice tasks

```
=== TASK 01: Vector add / sub / mul + reduction (format of 2026 midterm-1) ===

N = 2 << 20. Two int arrays a and b.
Fill:   a[i] = (i % 11) - 5
        b[i] = ((i * 7 + 3) % 11) - 5
Write three separate __device__ functions (add, subtract, multiply) and
compute c = a op b element-wise for each operation. Sum every c with the
reduction kernel. Implement the same on the host and verify.
Measure GPU vs CPU time of the multiplication.
Also print A, B and the three C arrays for N = 10 (like the real exam).

Checked output:
  Sum (addition) = ...
  Sum (subtraction) = ...
  Sum (multiplication) = ...

COMMON REQUIREMENTS (graded like the real midterm):
  - number of blocks = warp size * number of SMs (query both at runtime);
    threads per block = 256 (or another multiple of 32)
  - grid-stride loop in every kernel
  - input data filled ON THE GPU by a kernel, arrays on independent streams
  - async data migration: cudaMallocManaged + cudaMemPrefetchAsync
    (or cudaMalloc + cudaMemcpyAsync)
  - every sum is computed on the GPU with a shared-memory reduction kernel
  - __device__ function(s) for the per-element operation
  - host version of the computation + host verify function;
    print "Verify: PASSED" (or "Verify: FAILED")
  - print the hardware name; time the GPU kernel (cudaEvent) and the host
    version (std::chrono) and print both
  - free all memory, destroy streams/events; one file lastname.cu
  - use long long for sums
  - checked lines must look exactly like "Label = value"; if you print a
    label several times (e.g. small N first), the LAST occurrence is checked
```

```
=== TASK 02: Matrix multiplication (format of 2025 midterm-1) ===

N = 1 << 8. Two N x N int matrices, row-major, flat index i in [0, N*N).
Fill:   A[i] = (i * 3) % 10 - 4
        B[i] = (i * 5 + 1) % 10 - 5
C = A x B. One thread per element of C (grid-stride over N*N),
a __device__ function computes the dot product of row r of A and
column c of B. Sum all elements of C with the reduction kernel.
Host matmul + verify; time both. Print A, B, C for N = 4 first.
Bonus (not checked): find the largest N your GPU handles.

Checked output:
  Sum (C) = ...
  C[0][0] = ...
  C[N-1][N-1] = ...

COMMON REQUIREMENTS (graded like the real midterm):
  - number of blocks = warp size * number of SMs (query both at runtime);
    threads per block = 256 (or another multiple of 32)
  - grid-stride loop in every kernel
  - input data filled ON THE GPU by a kernel, arrays on independent streams
  - async data migration: cudaMallocManaged + cudaMemPrefetchAsync
    (or cudaMalloc + cudaMemcpyAsync)
  - every sum is computed on the GPU with a shared-memory reduction kernel
  - __device__ function(s) for the per-element operation
  - host version of the computation + host verify function;
    print "Verify: PASSED" (or "Verify: FAILED")
  - print the hardware name; time the GPU kernel (cudaEvent) and the host
    version (std::chrono) and print both
  - free all memory, destroy streams/events; one file lastname.cu
  - use long long for sums
  - checked lines must look exactly like "Label = value"; if you print a
    label several times (e.g. small N first), the LAST occurrence is checked
```

```
=== TASK 03: Integer SAXPY in place ===

N = 1 << 22, k = 3.
Fill:   x[i] = (i % 13) - 6
        y[i] = (i % 7) - 3
Compute y = k * x + y IN PLACE with a __device__ function saxpyOp.
Sum y with the reduction kernel.
Trap: the host check needs the ORIGINAL y - keep a copy before the kernel.

Checked output:
  Sum (y) = ...
  y[12345] = ...

COMMON REQUIREMENTS (graded like the real midterm):
  - number of blocks = warp size * number of SMs (query both at runtime);
    threads per block = 256 (or another multiple of 32)
  - grid-stride loop in every kernel
  - input data filled ON THE GPU by a kernel, arrays on independent streams
  - async data migration: cudaMallocManaged + cudaMemPrefetchAsync
    (or cudaMalloc + cudaMemcpyAsync)
  - every sum is computed on the GPU with a shared-memory reduction kernel
  - __device__ function(s) for the per-element operation
  - host version of the computation + host verify function;
    print "Verify: PASSED" (or "Verify: FAILED")
  - print the hardware name; time the GPU kernel (cudaEvent) and the host
    version (std::chrono) and print both
  - free all memory, destroy streams/events; one file lastname.cu
  - use long long for sums
  - checked lines must look exactly like "Label = value"; if you print a
    label several times (e.g. small N first), the LAST occurrence is checked
```

```
=== TASK 04: Dot product + max/min reduction ===

N = 1 << 21.
Fill:   a[i] = (i * 13 % 17) - 8
        b[i] = (i * 5 % 9) - 4
p[i] = a[i] * b[i] (device function).
Dot = sum of p (reduction). Max and Min of p with a max/min version of
the shared-memory reduction; combine block results with atomicMax /
atomicMin (or a second kernel pass).

Checked output:
  Dot = ...
  Max (a*b) = ...
  Min (a*b) = ...

COMMON REQUIREMENTS (graded like the real midterm):
  - number of blocks = warp size * number of SMs (query both at runtime);
    threads per block = 256 (or another multiple of 32)
  - grid-stride loop in every kernel
  - input data filled ON THE GPU by a kernel, arrays on independent streams
  - async data migration: cudaMallocManaged + cudaMemPrefetchAsync
    (or cudaMalloc + cudaMemcpyAsync)
  - every sum is computed on the GPU with a shared-memory reduction kernel
  - __device__ function(s) for the per-element operation
  - host version of the computation + host verify function;
    print "Verify: PASSED" (or "Verify: FAILED")
  - print the hardware name; time the GPU kernel (cudaEvent) and the host
    version (std::chrono) and print both
  - free all memory, destroy streams/events; one file lastname.cu
  - use long long for sums
  - checked lines must look exactly like "Label = value"; if you print a
    label several times (e.g. small N first), the LAST occurrence is checked
```

```
=== TASK 05: C = A + B transposed, sum and trace ===

N = 1024. N x N int matrices, row-major.
Fill:   A[r][c] = (r * 3 + c) % 10 - 5
        B[r][c] = (r + c * 2) % 9 - 4
C[r][c] = A[r][c] + B[c][r]   (grid-stride over N*N, r = i / N, c = i % N)
Sum all of C with the reduction kernel. Trace (C) = sum of the diagonal.

Checked output:
  Sum (C) = ...
  Trace (C) = ...

COMMON REQUIREMENTS (graded like the real midterm):
  - number of blocks = warp size * number of SMs (query both at runtime);
    threads per block = 256 (or another multiple of 32)
  - grid-stride loop in every kernel
  - input data filled ON THE GPU by a kernel, arrays on independent streams
  - async data migration: cudaMallocManaged + cudaMemPrefetchAsync
    (or cudaMalloc + cudaMemcpyAsync)
  - every sum is computed on the GPU with a shared-memory reduction kernel
  - __device__ function(s) for the per-element operation
  - host version of the computation + host verify function;
    print "Verify: PASSED" (or "Verify: FAILED")
  - print the hardware name; time the GPU kernel (cudaEvent) and the host
    version (std::chrono) and print both
  - free all memory, destroy streams/events; one file lastname.cu
  - use long long for sums
  - checked lines must look exactly like "Label = value"; if you print a
    label several times (e.g. small N first), the LAST occurrence is checked
```

```
=== TASK 06: Polynomial evaluation + counting with atomics ===

N = 1 << 20.
Fill:   x[i] = (i % 21) - 10
y[i] = 2x^3 - 3x^2 + x - 7, computed by a __device__ int poly(int x).
Sum (y) with the reduction kernel. Count of y > 0 with atomicAdd.

Checked output:
  Sum (y) = ...
  Count (y > 0) = ...

COMMON REQUIREMENTS (graded like the real midterm):
  - number of blocks = warp size * number of SMs (query both at runtime);
    threads per block = 256 (or another multiple of 32)
  - grid-stride loop in every kernel
  - input data filled ON THE GPU by a kernel, arrays on independent streams
  - async data migration: cudaMallocManaged + cudaMemPrefetchAsync
    (or cudaMalloc + cudaMemcpyAsync)
  - every sum is computed on the GPU with a shared-memory reduction kernel
  - __device__ function(s) for the per-element operation
  - host version of the computation + host verify function;
    print "Verify: PASSED" (or "Verify: FAILED")
  - print the hardware name; time the GPU kernel (cudaEvent) and the host
    version (std::chrono) and print both
  - free all memory, destroy streams/events; one file lastname.cu
  - use long long for sums
  - checked lines must look exactly like "Label = value"; if you print a
    label several times (e.g. small N first), the LAST occurrence is checked
```

```
=== TASK 07: Histogram with shared-memory privatization ===

N = 1 << 22.
Fill:   a[i] = ((i * 31 + 7) % 11) - 5      (values -5..5, 11 bins)
Each block builds its own histogram in __shared__ memory (atomicAdd on
shared), then adds it to the global histogram (atomicAdd on global).
Also Sum of a with the reduction kernel. Verify bins on the host.

Checked output (one line per bin, -5..5):
  Bin -5 = ...
  ...
  Bin 5 = ...
  Sum = ...

COMMON REQUIREMENTS (graded like the real midterm):
  - number of blocks = warp size * number of SMs (query both at runtime);
    threads per block = 256 (or another multiple of 32)
  - grid-stride loop in every kernel
  - input data filled ON THE GPU by a kernel, arrays on independent streams
  - async data migration: cudaMallocManaged + cudaMemPrefetchAsync
    (or cudaMalloc + cudaMemcpyAsync)
  - every sum is computed on the GPU with a shared-memory reduction kernel
  - __device__ function(s) for the per-element operation
  - host version of the computation + host verify function;
    print "Verify: PASSED" (or "Verify: FAILED")
  - print the hardware name; time the GPU kernel (cudaEvent) and the host
    version (std::chrono) and print both
  - free all memory, destroy streams/events; one file lastname.cu
  - use long long for sums
  - checked lines must look exactly like "Label = value"; if you print a
    label several times (e.g. small N first), the LAST occurrence is checked
```

```
=== TASK 08: Sum / Min / Max of pseudo-random data ===

N = 1 << 22. Use unsigned int arithmetic exactly like this:
        unsigned int h = ((unsigned int)i * 1103515245u + 12345u) >> 8;
        a[i] = (int)(h % 1001) - 500;
Compute Sum, Min and Max on the GPU with shared-memory reductions
(one kernel doing all three, or three kernels).

Checked output:
  Sum = ...
  Min = ...
  Max = ...

COMMON REQUIREMENTS (graded like the real midterm):
  - number of blocks = warp size * number of SMs (query both at runtime);
    threads per block = 256 (or another multiple of 32)
  - grid-stride loop in every kernel
  - input data filled ON THE GPU by a kernel, arrays on independent streams
  - async data migration: cudaMallocManaged + cudaMemPrefetchAsync
    (or cudaMalloc + cudaMemcpyAsync)
  - every sum is computed on the GPU with a shared-memory reduction kernel
  - __device__ function(s) for the per-element operation
  - host version of the computation + host verify function;
    print "Verify: PASSED" (or "Verify: FAILED")
  - print the hardware name; time the GPU kernel (cudaEvent) and the host
    version (std::chrono) and print both
  - free all memory, destroy streams/events; one file lastname.cu
  - use long long for sums
  - checked lines must look exactly like "Label = value"; if you print a
    label several times (e.g. small N first), the LAST occurrence is checked
```

```
=== TASK 09: 1D stencil ===

N = 1 << 21.
Fill:   a[i] = (i % 9) - 4
c[i] = a[i-1] + 2 * a[i] + a[i+1], out-of-range neighbours count as 0.
Use a __device__ function for the stencil. Sum c with the reduction kernel.
Bonus (not checked): load a tile + halo into __shared__ memory.

Checked output:
  Sum (c) = ...
  c[0] = ...
  c[N-1] = ...
  c[1000] = ...

COMMON REQUIREMENTS (graded like the real midterm):
  - number of blocks = warp size * number of SMs (query both at runtime);
    threads per block = 256 (or another multiple of 32)
  - grid-stride loop in every kernel
  - input data filled ON THE GPU by a kernel, arrays on independent streams
  - async data migration: cudaMallocManaged + cudaMemPrefetchAsync
    (or cudaMalloc + cudaMemcpyAsync)
  - every sum is computed on the GPU with a shared-memory reduction kernel
  - __device__ function(s) for the per-element operation
  - host version of the computation + host verify function;
    print "Verify: PASSED" (or "Verify: FAILED")
  - print the hardware name; time the GPU kernel (cudaEvent) and the host
    version (std::chrono) and print both
  - free all memory, destroy streams/events; one file lastname.cu
  - use long long for sums
  - checked lines must look exactly like "Label = value"; if you print a
    label several times (e.g. small N first), the LAST occurrence is checked
```

```
=== TASK 10: Matrix-vector multiplication ===

N = 2048. A is N x N row-major, x has N elements.
Fill:   A[r][c] = (r * 7 + c * 3) % 11 - 5     (fill A and x on different streams)
        x[c] = (c % 5) - 2
y = A * x, one thread per row (grid-stride over rows), dot product in a
__device__ function. Sum (y) with the reduction kernel, Max (y) with a
max reduction.

Checked output:
  Sum (y) = ...
  Max (y) = ...
  y[7] = ...

COMMON REQUIREMENTS (graded like the real midterm):
  - number of blocks = warp size * number of SMs (query both at runtime);
    threads per block = 256 (or another multiple of 32)
  - grid-stride loop in every kernel
  - input data filled ON THE GPU by a kernel, arrays on independent streams
  - async data migration: cudaMallocManaged + cudaMemPrefetchAsync
    (or cudaMalloc + cudaMemcpyAsync)
  - every sum is computed on the GPU with a shared-memory reduction kernel
  - __device__ function(s) for the per-element operation
  - host version of the computation + host verify function;
    print "Verify: PASSED" (or "Verify: FAILED")
  - print the hardware name; time the GPU kernel (cudaEvent) and the host
    version (std::chrono) and print both
  - free all memory, destroy streams/events; one file lastname.cu
  - use long long for sums
  - checked lines must look exactly like "Label = value"; if you print a
    label several times (e.g. small N first), the LAST occurrence is checked
```

