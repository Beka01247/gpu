#include <cstdio>
#include <cstdlib>
#include <climits>
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

const int THREADS = 256;

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

__device__ int mulDevice(int x, int y) { return x * y; }

__global__ void fillA(int *a, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        a[i] = (i * 13 % 17) - 8;
}

__global__ void fillB(int *b, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        b[i] = (i * 5 % 9) - 4;
}

__global__ void productKernel(const int *a, const int *b, int *p, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        p[i] = mulDevice(a[i], b[i]);
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

__global__ void reduceMax(const int *in, int *result, int n)
{
    __shared__ int cache[THREADS];
    int tid = threadIdx.x;
    int m = INT_MIN;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        m = max(m, in[i]);
    cache[tid] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            cache[tid] = max(cache[tid], cache[tid + s]);
        __syncthreads();
    }
    if (tid == 0)
        atomicMax(result, cache[0]);
}

__global__ void reduceMin(const int *in, int *result, int n)
{
    __shared__ int cache[THREADS];
    int tid = threadIdx.x;
    int m = INT_MAX;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        m = min(m, in[i]);
    cache[tid] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            cache[tid] = min(cache[tid], cache[tid + s]);
        __syncthreads();
    }
    if (tid == 0)
        atomicMin(result, cache[0]);
}

int main()
{
    const int N = 1 << 21;
    size_t bytes = N * sizeof(int);

    int device;
    CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device));
    int blocks = prop.warpSize * prop.multiProcessorCount;
    printf("Hardware name: %s\n", prop.name);

    int *a, *b, *p, *mx, *mn;
    long long *dot;
    CHECK(cudaMallocManaged(&a, bytes));
    CHECK(cudaMallocManaged(&b, bytes));
    CHECK(cudaMallocManaged(&p, bytes));
    CHECK(cudaMallocManaged(&dot, sizeof(long long)));
    CHECK(cudaMallocManaged(&mx, sizeof(int)));
    CHECK(cudaMallocManaged(&mn, sizeof(int)));
    *dot = 0;
    *mx = INT_MIN;
    *mn = INT_MAX;

    cudaStream_t sA, sB;
    CHECK(cudaStreamCreate(&sA));
    CHECK(cudaStreamCreate(&sB));
    prefetch(a, bytes, device, sA);
    prefetch(b, bytes, device, sB);
    prefetch(p, bytes, device);

    fillA<<<blocks, THREADS, 0, sA>>>(a, N);
    fillB<<<blocks, THREADS, 0, sB>>>(b, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    productKernel<<<blocks, THREADS>>>(a, b, p, N);
    reduceSum<<<blocks, THREADS>>>(p, dot, N);
    reduceMax<<<blocks, THREADS>>>(p, mx, N);
    reduceMin<<<blocks, THREADS>>>(p, mn, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    float gpuMs = 0;
    CHECK(cudaEventElapsedTime(&gpuMs, start, stop));

    prefetch(a, bytes, cudaCpuDeviceId);
    prefetch(b, bytes, cudaCpuDeviceId);
    CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    long long cpuDot = 0;
    int cpuMax = INT_MIN, cpuMin = INT_MAX;
    for (int i = 0; i < N; i++) {
        int v = a[i] * b[i];
        cpuDot += v;
        cpuMax = v > cpuMax ? v : cpuMax;
        cpuMin = v < cpuMin ? v : cpuMin;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("Dot = %lld\n", *dot);
    printf("Max (a*b) = %d\n", *mx);
    printf("Min (a*b) = %d\n", *mn);
    printf("Time: CPU %.3f ms, GPU %.3f ms\n", cpuMs, gpuMs);
    printf("Verify: %s\n", *dot == cpuDot && *mx == cpuMax && *mn == cpuMin ? "PASSED" : "FAILED");

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaStreamDestroy(sA));
    CHECK(cudaStreamDestroy(sB));
    CHECK(cudaFree(a));
    CHECK(cudaFree(b));
    CHECK(cudaFree(p));
    CHECK(cudaFree(dot));
    CHECK(cudaFree(mx));
    CHECK(cudaFree(mn));
    return 0;
}
