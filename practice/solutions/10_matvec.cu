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

__global__ void fillMatrix(int *m, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * n; i += blockDim.x * gridDim.x) {
        int r = i / n, c = i % n;
        m[i] = (r * 7 + c * 3) % 11 - 5;
    }
}

__global__ void fillVector(int *x, int n)
{
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < n; c += blockDim.x * gridDim.x)
        x[c] = (c % 5) - 2;
}

__device__ int dotRow(const int *m, const int *x, int n, int row)
{
    int sum = 0;
    for (int c = 0; c < n; c++)
        sum += m[row * n + c] * x[c];
    return sum;
}

__global__ void matVec(const int *m, const int *x, int *y, int n)
{
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += blockDim.x * gridDim.x)
        y[row] = dotRow(m, x, n, row);
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

int main()
{
    const int N = 2048;
    size_t matBytes = (size_t)N * N * sizeof(int);
    size_t vecBytes = N * sizeof(int);

    int device;
    CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device));
    int blocks = prop.warpSize * prop.multiProcessorCount;
    printf("Hardware name: %s\n", prop.name);

    int *m, *x, *y, *mx;
    long long *sum;
    CHECK(cudaMallocManaged(&m, matBytes));
    CHECK(cudaMallocManaged(&x, vecBytes));
    CHECK(cudaMallocManaged(&y, vecBytes));
    CHECK(cudaMallocManaged(&mx, sizeof(int)));
    CHECK(cudaMallocManaged(&sum, sizeof(long long)));
    *mx = INT_MIN;
    *sum = 0;

    cudaStream_t sM, sX;
    CHECK(cudaStreamCreate(&sM));
    CHECK(cudaStreamCreate(&sX));
    prefetch(m, matBytes, device, sM);
    prefetch(x, vecBytes, device, sX);
    prefetch(y, vecBytes, device);

    fillMatrix<<<blocks, THREADS, 0, sM>>>(m, N);
    fillVector<<<blocks, THREADS, 0, sX>>>(x, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    matVec<<<blocks, THREADS>>>(m, x, y, N);
    CHECK(cudaEventRecord(stop));
    reduceSum<<<blocks, THREADS>>>(y, sum, N);
    reduceMax<<<blocks, THREADS>>>(y, mx, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    float gpuMs = 0;
    CHECK(cudaEventElapsedTime(&gpuMs, start, stop));

    prefetch(m, matBytes, cudaCpuDeviceId);
    prefetch(x, vecBytes, cudaCpuDeviceId);
    prefetch(y, vecBytes, cudaCpuDeviceId);
    CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = true;
    long long cpuSum = 0;
    int cpuMax = INT_MIN;
    for (int r = 0; r < N; r++) {
        int v = 0;
        for (int c = 0; c < N; c++)
            v += m[r * N + c] * x[c];
        ok = ok && y[r] == v;
        cpuSum += v;
        cpuMax = v > cpuMax ? v : cpuMax;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("Sum (y) = %lld\n", *sum);
    printf("Max (y) = %d\n", *mx);
    printf("y[7] = %d\n", y[7]);
    printf("Time: CPU %.3f ms, GPU %.3f ms\n", cpuMs, gpuMs);
    printf("Verify: %s\n", ok && *sum == cpuSum && *mx == cpuMax ? "PASSED" : "FAILED");

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaStreamDestroy(sM));
    CHECK(cudaStreamDestroy(sX));
    CHECK(cudaFree(m));
    CHECK(cudaFree(x));
    CHECK(cudaFree(y));
    CHECK(cudaFree(mx));
    CHECK(cudaFree(sum));
    return 0;
}
