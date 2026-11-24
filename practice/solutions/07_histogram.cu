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

const int THREADS = 256;
const int BINS = 11;

__device__ int binOf(int v) { return v + 5; }

__global__ void fillA(int *a, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        a[i] = ((i * 31 + 7) % 11) - 5;
}

__global__ void histogram(const int *a, int *hist, int n)
{
    __shared__ int local[BINS];
    for (int j = threadIdx.x; j < BINS; j += blockDim.x)
        local[j] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        atomicAdd(&local[binOf(a[i])], 1);
    __syncthreads();
    for (int j = threadIdx.x; j < BINS; j += blockDim.x)
        atomicAdd(&hist[j], local[j]);
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
    const int N = 1 << 22;
    size_t bytes = N * sizeof(int);

    int device;
    CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device));
    int blocks = prop.warpSize * prop.multiProcessorCount;
    printf("Hardware name: %s\n", prop.name);

    int *a, *hist;
    long long *sum;
    CHECK(cudaMallocManaged(&a, bytes));
    CHECK(cudaMallocManaged(&hist, BINS * sizeof(int)));
    CHECK(cudaMallocManaged(&sum, sizeof(long long)));
    for (int j = 0; j < BINS; j++)
        hist[j] = 0;
    *sum = 0;

    cudaStream_t s1, s2;
    CHECK(cudaStreamCreate(&s1));
    CHECK(cudaStreamCreate(&s2));
    CHECK(cudaMemPrefetchAsync(a, bytes, device, s1));
    CHECK(cudaMemPrefetchAsync(hist, BINS * sizeof(int), device, s2));

    fillA<<<blocks, THREADS, 0, s1>>>(a, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start, s1));
    histogram<<<blocks, THREADS, 0, s1>>>(a, hist, N);
    CHECK(cudaEventRecord(stop, s1));
    reduceSum<<<blocks, THREADS, 0, s2>>>(a, sum, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    float gpuMs = 0;
    CHECK(cudaEventElapsedTime(&gpuMs, start, stop));

    CHECK(cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId));
    CHECK(cudaMemPrefetchAsync(hist, BINS * sizeof(int), cudaCpuDeviceId));
    CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    int cpuHist[BINS] = {0};
    long long cpuSum = 0;
    for (int i = 0; i < N; i++) {
        cpuHist[a[i] + 5]++;
        cpuSum += a[i];
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    bool ok = *sum == cpuSum;
    for (int j = 0; j < BINS; j++) {
        printf("Bin %d = %d\n", j - 5, hist[j]);
        ok = ok && hist[j] == cpuHist[j];
    }
    printf("Sum = %lld\n", *sum);
    printf("Time: CPU %.3f ms, GPU %.3f ms\n", cpuMs, gpuMs);
    printf("Verify: %s\n", ok ? "PASSED" : "FAILED");

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaStreamDestroy(s1));
    CHECK(cudaStreamDestroy(s2));
    CHECK(cudaFree(a));
    CHECK(cudaFree(hist));
    CHECK(cudaFree(sum));
    return 0;
}
