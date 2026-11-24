#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CHECK(call)                                                       \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            printf("CUDA error: %s (%s:%d)\n", cudaGetErrorString(err),   \
                   __FILE__, __LINE__);                                   \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

// multiple of warp size, power of two for the reduction
const int THREADS = 256;

// ============================ shared kernels ============================

__global__ void initRng(curandState *states, int count, unsigned long long seed)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
        curand_init(seed, i, 0, &states[i]);
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

// ======================== TASK 1: Monte Carlo Pi ========================

__global__ void monteCarloPi(curandState *states, int samplesPerThread, float *out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local = states[tid];
    int acc = 0;
    for (int i = 0; i < samplesPerThread; i++) {
        float x = curand_uniform(&local);
        float y = curand_uniform(&local);
        if (x * x + y * y < 1.0f)
            acc++;
    }
    states[tid] = local;
    out[tid] = 4.0f * acc / samplesPerThread;
}

void task1()
{
    const int nsamples = 10000000;
    const int threadsPerBlock = 128;
    const int blocks = 32;
    const int gridSize = threadsPerBlock * blocks;
    const int samplesPerThread = nsamples / gridSize;

    curandState *states;
    float *out;
    CHECK(cudaMalloc(&states, gridSize * sizeof(curandState)));
    CHECK(cudaMallocManaged(&out, gridSize * sizeof(float)));

    initRng<<<blocks, threadsPerBlock>>>(states, gridSize, 1);
    monteCarloPi<<<blocks, threadsPerBlock>>>(states, samplesPerThread, out);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    double mean = 0;
    for (int i = 0; i < gridSize; i++)
        mean += out[i];
    printf("TASK 1: pi = %.6f\n\n", mean / gridSize);

    CHECK(cudaFree(states));
    CHECK(cudaFree(out));
}

// ===================== TASK 2: Matrix multiplication =====================

__global__ void generateMatrices(curandState *states, int *a, int *b, int *c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local = states[tid];
    for (int i = tid; i < n * n; i += blockDim.x * gridDim.x) {
        a[i] = curand(&local) % 10;
        b[i] = curand(&local) % 10;
        c[i] = 0;
    }
    states[tid] = local;
}

__device__ int dotRowCol(const int *a, const int *b, int n, int row, int col)
{
    int sum = 0;
    for (int k = 0; k < n; k++)
        sum += a[row * n + k] * b[k * n + col];
    return sum;
}

__global__ void matMulKernel(const int *a, const int *b, int *c, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * n; i += blockDim.x * gridDim.x)
        c[i] = dotRowCol(a, b, n, i / n, i % n);
}

void matMulHost(const int *a, const int *b, int *c, int n)
{
    for (int row = 0; row < n; row++)
        for (int col = 0; col < n; col++) {
            int sum = 0;
            for (int k = 0; k < n; k++)
                sum += a[row * n + k] * b[k * n + col];
            c[row * n + col] = sum;
        }
}

bool checkMultiplication(const int *gpu, const int *cpu, int n)
{
    for (int i = 0; i < n * n; i++)
        if (gpu[i] != cpu[i])
            return false;
    return true;
}

long long sumHost(const int *c, int n)
{
    long long s = 0;
    for (int i = 0; i < n * n; i++)
        s += c[i];
    return s;
}

void printMatrix(const char *name, const int *m, int n)
{
    printf("%s =\n", name);
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++)
            printf("%4d ", m[r * n + c]);
        printf("\n");
    }
}

void runMatMul(int n, bool show, int device, int blocks, curandState *states)
{
    size_t bytes = (size_t)n * n * sizeof(int);
    int *a, *b, *c, *cHost;
    long long *gpuSum;
    CHECK(cudaMallocManaged(&a, bytes));
    CHECK(cudaMallocManaged(&b, bytes));
    CHECK(cudaMallocManaged(&c, bytes));
    CHECK(cudaMallocManaged(&gpuSum, sizeof(long long)));
    cHost = (int *)malloc(bytes);
    *gpuSum = 0;

    CHECK(cudaMemPrefetchAsync(a, bytes, device));
    CHECK(cudaMemPrefetchAsync(b, bytes, device));
    CHECK(cudaMemPrefetchAsync(c, bytes, device));
    CHECK(cudaMemPrefetchAsync(gpuSum, sizeof(long long), device));

    generateMatrices<<<blocks, THREADS>>>(states, a, b, c, n);

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    matMulKernel<<<blocks, THREADS>>>(a, b, c, n);
    CHECK(cudaEventRecord(stop));
    reduceSum<<<blocks, THREADS>>>(c, gpuSum, n * n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    float gpuMs = 0;
    CHECK(cudaEventElapsedTime(&gpuMs, start, stop));

    CHECK(cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId));
    CHECK(cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId));
    CHECK(cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId));
    CHECK(cudaMemPrefetchAsync(gpuSum, sizeof(long long), cudaCpuDeviceId));
    CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    matMulHost(a, b, cHost, n);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    long long cpuSum = sumHost(cHost, n);

    if (show) {
        printf("Generated arrays for N = %d:\n", n);
        printMatrix("A", a, n);
        printMatrix("B", b, n);
        printf("CPU results:\n");
        printMatrix("C", cHost, n);
        printf("Sum = %lld\n", cpuSum);
        printf("GPU results:\n");
        printMatrix("C", c, n);
        printf("Sum = %lld\n", *gpuSum);
    }
    printf("N = %d: multiplication %s, reduction %s, GPU %.3f ms, CPU %.3f ms\n", n,
           checkMultiplication(c, cHost, n) ? "PASSED" : "FAILED",
           *gpuSum == cpuSum ? "PASSED" : "FAILED", gpuMs, cpuMs);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(a));
    CHECK(cudaFree(b));
    CHECK(cudaFree(c));
    CHECK(cudaFree(gpuSum));
    free(cHost);
}

int largestN()
{
    size_t freeBytes, totalBytes;
    CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    // three int matrices of N x N have to fit into free memory
    return (int)sqrt((double)freeBytes / (3 * sizeof(int)));
}

void task2()
{
    int device;
    CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device));
    int blocks = prop.warpSize * prop.multiProcessorCount;
    printf("Name: %s\n", prop.name);

    curandState *states;
    CHECK(cudaMalloc(&states, blocks * THREADS * sizeof(curandState)));
    initRng<<<blocks, THREADS>>>(states, blocks * THREADS, 42);
    CHECK(cudaGetLastError());

    runMatMul(3, true, device, blocks, states);
    runMatMul(1 << 9, false, device, blocks, states);
    printf("Largest array size: N = %d\n", largestN());

    CHECK(cudaFree(states));
}

int main()
{
    task1();
    task2();
    return 0;
}
