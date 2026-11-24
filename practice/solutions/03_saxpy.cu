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

__device__ int saxpyOp(int k, int x, int y) { return k * x + y; }

__global__ void fillX(int *x, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        x[i] = (i % 13) - 6;
}

__global__ void fillY(int *y, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        y[i] = (i % 7) - 3;
}

__global__ void saxpyKernel(int k, const int *x, int *y, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        y[i] = saxpyOp(k, x[i], y[i]);
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

void saxpyHost(int k, const int *x, int *y, int n)
{
    for (int i = 0; i < n; i++)
        y[i] = k * x[i] + y[i];
}

int main()
{
    const int N = 1 << 22;
    const int K = 3;
    size_t bytes = N * sizeof(int);

    int device;
    CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device));
    int blocks = prop.warpSize * prop.multiProcessorCount;
    printf("Hardware name: %s\n", prop.name);

    int *x, *y, *yHost;
    long long *sum;
    CHECK(cudaMallocManaged(&x, bytes));
    CHECK(cudaMallocManaged(&y, bytes));
    CHECK(cudaMallocManaged(&sum, sizeof(long long)));
    yHost = (int *)malloc(bytes);
    *sum = 0;

    cudaStream_t sX, sY;
    CHECK(cudaStreamCreate(&sX));
    CHECK(cudaStreamCreate(&sY));
    CHECK(cudaMemPrefetchAsync(x, bytes, device, sX));
    CHECK(cudaMemPrefetchAsync(y, bytes, device, sY));
    CHECK(cudaMemPrefetchAsync(sum, sizeof(long long), device));

    fillX<<<blocks, THREADS, 0, sX>>>(x, N);
    fillY<<<blocks, THREADS, 0, sY>>>(y, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // keep the original y for the host check: the kernel overwrites it
    CHECK(cudaMemcpy(yHost, y, bytes, cudaMemcpyDeviceToHost));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    saxpyKernel<<<blocks, THREADS>>>(K, x, y, N);
    CHECK(cudaEventRecord(stop));
    reduceSum<<<blocks, THREADS>>>(y, sum, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    float gpuMs = 0;
    CHECK(cudaEventElapsedTime(&gpuMs, start, stop));

    CHECK(cudaMemPrefetchAsync(x, bytes, cudaCpuDeviceId));
    CHECK(cudaMemPrefetchAsync(y, bytes, cudaCpuDeviceId));
    CHECK(cudaMemPrefetchAsync(sum, sizeof(long long), cudaCpuDeviceId));
    CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    saxpyHost(K, x, yHost, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    bool ok = true;
    long long cpuSum = 0;
    for (int i = 0; i < N; i++) {
        ok = ok && y[i] == yHost[i];
        cpuSum += yHost[i];
    }

    printf("Sum (y) = %lld\n", *sum);
    printf("y[12345] = %d\n", y[12345]);
    printf("Time: CPU %.3f ms, GPU %.3f ms\n", cpuMs, gpuMs);
    printf("Verify: %s\n", ok && *sum == cpuSum ? "PASSED" : "FAILED");

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaStreamDestroy(sX));
    CHECK(cudaStreamDestroy(sY));
    CHECK(cudaFree(x));
    CHECK(cudaFree(y));
    CHECK(cudaFree(sum));
    free(yHost);
    return 0;
}
