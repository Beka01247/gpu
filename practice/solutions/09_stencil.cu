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

__host__ __device__ int stencilAt(const int *a, int i, int n)
{
    int left = i > 0 ? a[i - 1] : 0;
    int right = i < n - 1 ? a[i + 1] : 0;
    return left + 2 * a[i] + right;
}

__global__ void fillA(int *a, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        a[i] = (i % 9) - 4;
}

__global__ void stencilKernel(const int *a, int *c, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        c[i] = stencilAt(a, i, n);
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
    const int N = 1 << 21;
    size_t bytes = N * sizeof(int);

    int device;
    CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device));
    int blocks = prop.warpSize * prop.multiProcessorCount;
    printf("Hardware name: %s\n", prop.name);

    int *a, *c, *cHost;
    long long *sum;
    CHECK(cudaMallocManaged(&a, bytes));
    CHECK(cudaMallocManaged(&c, bytes));
    CHECK(cudaMallocManaged(&sum, sizeof(long long)));
    cHost = (int *)malloc(bytes);
    *sum = 0;

    cudaStream_t s1, s2;
    CHECK(cudaStreamCreate(&s1));
    CHECK(cudaStreamCreate(&s2));
    CHECK(cudaMemPrefetchAsync(a, bytes, device, s1));
    CHECK(cudaMemPrefetchAsync(c, bytes, device, s2));

    fillA<<<blocks, THREADS, 0, s1>>>(a, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    stencilKernel<<<blocks, THREADS>>>(a, c, N);
    CHECK(cudaEventRecord(stop));
    reduceSum<<<blocks, THREADS>>>(c, sum, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    float gpuMs = 0;
    CHECK(cudaEventElapsedTime(&gpuMs, start, stop));

    CHECK(cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId));
    CHECK(cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId));
    CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++)
        cHost[i] = stencilAt(a, i, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    bool ok = true;
    long long cpuSum = 0;
    for (int i = 0; i < N; i++) {
        ok = ok && c[i] == cHost[i];
        cpuSum += cHost[i];
    }

    printf("Sum (c) = %lld\n", *sum);
    printf("c[0] = %d\n", c[0]);
    printf("c[N-1] = %d\n", c[N - 1]);
    printf("c[1000] = %d\n", c[1000]);
    printf("Time: CPU %.3f ms, GPU %.3f ms\n", cpuMs, gpuMs);
    printf("Verify: %s\n", ok && *sum == cpuSum ? "PASSED" : "FAILED");

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaStreamDestroy(s1));
    CHECK(cudaStreamDestroy(s2));
    CHECK(cudaFree(a));
    CHECK(cudaFree(c));
    CHECK(cudaFree(sum));
    free(cHost);
    return 0;
}
