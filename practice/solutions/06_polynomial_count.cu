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

__device__ int poly(int x) { return 2 * x * x * x - 3 * x * x + x - 7; }

__global__ void fillX(int *x, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        x[i] = (i % 21) - 10;
}

__global__ void polyKernel(const int *x, int *y, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        y[i] = poly(x[i]);
}

__global__ void countPositive(const int *y, int *count, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        if (y[i] > 0)
            atomicAdd(count, 1);
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
    const int N = 1 << 20;
    size_t bytes = N * sizeof(int);

    int device;
    CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device));
    int blocks = prop.warpSize * prop.multiProcessorCount;
    printf("Hardware name: %s\n", prop.name);

    int *x, *y, *count;
    long long *sum;
    CHECK(cudaMallocManaged(&x, bytes));
    CHECK(cudaMallocManaged(&y, bytes));
    CHECK(cudaMallocManaged(&count, sizeof(int)));
    CHECK(cudaMallocManaged(&sum, sizeof(long long)));
    *count = 0;
    *sum = 0;

    cudaStream_t s1, s2;
    CHECK(cudaStreamCreate(&s1));
    CHECK(cudaStreamCreate(&s2));
    CHECK(cudaMemPrefetchAsync(x, bytes, device, s1));
    CHECK(cudaMemPrefetchAsync(y, bytes, device, s2));

    fillX<<<blocks, THREADS, 0, s1>>>(x, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    polyKernel<<<blocks, THREADS>>>(x, y, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaDeviceSynchronize());
    reduceSum<<<blocks, THREADS, 0, s1>>>(y, sum, N);
    countPositive<<<blocks, THREADS, 0, s2>>>(y, count, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    float gpuMs = 0;
    CHECK(cudaEventElapsedTime(&gpuMs, start, stop));

    CHECK(cudaMemPrefetchAsync(x, bytes, cudaCpuDeviceId));
    CHECK(cudaMemPrefetchAsync(y, bytes, cudaCpuDeviceId));
    CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = true;
    long long cpuSum = 0;
    int cpuCount = 0;
    for (int i = 0; i < N; i++) {
        int v = 2 * x[i] * x[i] * x[i] - 3 * x[i] * x[i] + x[i] - 7;
        ok = ok && y[i] == v;
        cpuSum += v;
        cpuCount += v > 0;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("Sum (y) = %lld\n", *sum);
    printf("Count (y > 0) = %d\n", *count);
    printf("Time: CPU %.3f ms, GPU %.3f ms\n", cpuMs, gpuMs);
    printf("Verify: %s\n", ok && *sum == cpuSum && *count == cpuCount ? "PASSED" : "FAILED");

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaStreamDestroy(s1));
    CHECK(cudaStreamDestroy(s2));
    CHECK(cudaFree(x));
    CHECK(cudaFree(y));
    CHECK(cudaFree(count));
    CHECK(cudaFree(sum));
    return 0;
}
