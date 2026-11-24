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

__host__ __device__ int pseudoRandom(int i)
{
    unsigned int h = ((unsigned int)i * 1103515245u + 12345u) >> 8;
    return (int)(h % 1001) - 500;
}

__global__ void fillA(int *a, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        a[i] = pseudoRandom(i);
}

// one pass computes sum, min and max together
__global__ void reduceAll(const int *in, long long *sum, int *mn, int *mx, int n)
{
    __shared__ long long cs[THREADS];
    __shared__ int cmin[THREADS];
    __shared__ int cmax[THREADS];
    int tid = threadIdx.x;

    long long s = 0;
    int lo = INT_MAX, hi = INT_MIN;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        s += in[i];
        lo = min(lo, in[i]);
        hi = max(hi, in[i]);
    }
    cs[tid] = s;
    cmin[tid] = lo;
    cmax[tid] = hi;
    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            cs[tid] += cs[tid + step];
            cmin[tid] = min(cmin[tid], cmin[tid + step]);
            cmax[tid] = max(cmax[tid], cmax[tid + step]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd((unsigned long long *)sum, (unsigned long long)cs[0]);
        atomicMin(mn, cmin[0]);
        atomicMax(mx, cmax[0]);
    }
}

int main()
{
    const int N = 1 << 22;
    size_t bytes = N * sizeof(int);

    int device;
    CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device));
    int blocks = prop.warpSize * prop.multiProcessorCount;
    printf("Hardware name: %s\n", prop.name);

    int *a, *mn, *mx;
    long long *sum;
    CHECK(cudaMallocManaged(&a, bytes));
    CHECK(cudaMallocManaged(&sum, sizeof(long long)));
    CHECK(cudaMallocManaged(&mn, sizeof(int)));
    CHECK(cudaMallocManaged(&mx, sizeof(int)));
    *sum = 0;
    *mn = INT_MAX;
    *mx = INT_MIN;

    cudaStream_t s1;
    CHECK(cudaStreamCreate(&s1));
    CHECK(cudaMemPrefetchAsync(a, bytes, device, s1));

    fillA<<<blocks, THREADS, 0, s1>>>(a, N);
    CHECK(cudaGetLastError());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start, s1));
    reduceAll<<<blocks, THREADS, 0, s1>>>(a, sum, mn, mx, N);
    CHECK(cudaEventRecord(stop, s1));
    CHECK(cudaGetLastError());
    CHECK(cudaStreamSynchronize(s1));
    float gpuMs = 0;
    CHECK(cudaEventElapsedTime(&gpuMs, start, stop));

    CHECK(cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId));
    CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = true;
    long long cpuSum = 0;
    int cpuMin = INT_MAX, cpuMax = INT_MIN;
    for (int i = 0; i < N; i++) {
        int v = pseudoRandom(i);
        ok = ok && a[i] == v;
        cpuSum += v;
        cpuMin = v < cpuMin ? v : cpuMin;
        cpuMax = v > cpuMax ? v : cpuMax;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("Sum = %lld\n", *sum);
    printf("Min = %d\n", *mn);
    printf("Max = %d\n", *mx);
    printf("Time: CPU %.3f ms, GPU %.3f ms\n", cpuMs, gpuMs);
    printf("Verify: %s\n", ok && *sum == cpuSum && *mn == cpuMin && *mx == cpuMax ? "PASSED" : "FAILED");

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaStreamDestroy(s1));
    CHECK(cudaFree(a));
    CHECK(cudaFree(sum));
    CHECK(cudaFree(mn));
    CHECK(cudaFree(mx));
    return 0;
}
