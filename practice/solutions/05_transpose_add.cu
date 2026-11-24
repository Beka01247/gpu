#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#define CHECK(call)                                                       \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            printf("CUDA error: %s (%s:%d)\n", cudaGetErrorString(err),   \
                   __FILE__, __LINE__);                                   \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

// skipped where managed memory has only limited support (Windows, WSL)
void prefetch(const void *ptr, size_t bytes, int location, cudaStream_t stream = 0)
{
    int device, concurrent;
    CHECK(cudaGetDevice(&device));
    CHECK(cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, device));
    if (!concurrent)
        return;
#if CUDART_VERSION >= 13000
    cudaMemLocation loc;
    loc.type = location == cudaCpuDeviceId ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
    loc.id = location == cudaCpuDeviceId ? 0 : location;
    CHECK(cudaMemPrefetchAsync(ptr, bytes, loc, 0, stream));
#else
    CHECK(cudaMemPrefetchAsync(ptr, bytes, location, stream));
#endif
}

const int THREADS = 256;

__global__ void fillA(int *m, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * n; i += blockDim.x * gridDim.x) {
        int r = i / n, c = i % n;
        m[i] = (r * 3 + c) % 10 - 5;
    }
}

__global__ void fillB(int *m, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * n; i += blockDim.x * gridDim.x) {
        int r = i / n, c = i % n;
        m[i] = (r + c * 2) % 9 - 4;
    }
}

__device__ int addTransposed(const int *a, const int *b, int n, int r, int c)
{
    return a[r * n + c] + b[c * n + r];
}

__global__ void addTransposeKernel(const int *a, const int *b, int *c, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * n; i += blockDim.x * gridDim.x)
        c[i] = addTransposed(a, b, n, i / n, i % n);
}

__global__ void diagonalKernel(const int *c, int *diag, int n)
{
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += blockDim.x * gridDim.x)
        diag[k] = c[k * n + k];
}

__global__ void reduceSum(const int *in, long long *result, int n)
{
    __shared__ long long cache[THREADS];
    int tid = threadIdx.x;
    long long sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        sum += in[i];
    cache[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            cache[tid] += cache[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd((unsigned long long *)result, (unsigned long long)cache[0]);
}

int main()
{
    const int N = 1024;
    size_t bytes = (size_t)N * N * sizeof(int);

    int device;
    CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device));
    int blocks = prop.warpSize * prop.multiProcessorCount;
    printf("Hardware name: %s\n", prop.name);

    int *a, *b, *c, *diag;
    long long *sums;
    CHECK(cudaMallocManaged(&a, bytes));
    CHECK(cudaMallocManaged(&b, bytes));
    CHECK(cudaMallocManaged(&c, bytes));
    CHECK(cudaMallocManaged(&diag, N * sizeof(int)));
    CHECK(cudaMallocManaged(&sums, 2 * sizeof(long long)));
    sums[0] = 0;
    sums[1] = 0;

    cudaStream_t sA, sB;
    CHECK(cudaStreamCreate(&sA));
    CHECK(cudaStreamCreate(&sB));
    prefetch(a, bytes, device, sA);
    prefetch(b, bytes, device, sB);
    prefetch(c, bytes, device);

    fillA<<<blocks, THREADS, 0, sA>>>(a, N);
    fillB<<<blocks, THREADS, 0, sB>>>(b, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    addTransposeKernel<<<blocks, THREADS>>>(a, b, c, N);
    CHECK(cudaEventRecord(stop));
    diagonalKernel<<<blocks, THREADS>>>(c, diag, N);
    reduceSum<<<blocks, THREADS>>>(c, &sums[0], N * N);
    reduceSum<<<blocks, THREADS>>>(diag, &sums[1], N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    float gpuMs = 0;
    CHECK(cudaEventElapsedTime(&gpuMs, start, stop));

    prefetch(a, bytes, cudaCpuDeviceId);
    prefetch(b, bytes, cudaCpuDeviceId);
    prefetch(c, bytes, cudaCpuDeviceId);
    CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = true;
    long long cpuSum = 0, cpuTrace = 0;
    for (int r = 0; r < N; r++)
        for (int col = 0; col < N; col++) {
            int v = a[r * N + col] + b[col * N + r];
            ok = ok && c[r * N + col] == v;
            cpuSum += v;
            if (r == col)
                cpuTrace += v;
        }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("Sum (C) = %lld\n", sums[0]);
    printf("Trace (C) = %lld\n", sums[1]);
    printf("Time: CPU %.3f ms, GPU %.3f ms\n", cpuMs, gpuMs);
    printf("Verify: %s\n", ok && sums[0] == cpuSum && sums[1] == cpuTrace ? "PASSED" : "FAILED");

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaStreamDestroy(sA));
    CHECK(cudaStreamDestroy(sB));
    CHECK(cudaFree(a));
    CHECK(cudaFree(b));
    CHECK(cudaFree(c));
    CHECK(cudaFree(diag));
    CHECK(cudaFree(sums));
    return 0;
}
