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

__global__ void fillA(int *m, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * n; i += blockDim.x * gridDim.x)
        m[i] = (i * 3) % 10 - 4;
}

__global__ void fillB(int *m, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * n; i += blockDim.x * gridDim.x)
        m[i] = (i * 5 + 1) % 10 - 5;
}

__device__ int dotRowCol(const int *a, const int *b, int n, int row, int col)
{
    int sum = 0;
    for (int k = 0; k < n; k++)
        sum += a[row * n + k] * b[k * n + col];
    return sum;
}

__global__ void matMul(const int *a, const int *b, int *c, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * n; i += blockDim.x * gridDim.x)
        c[i] = dotRowCol(a, b, n, i / n, i % n);
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

void matMulHost(const int *a, const int *b, int *c, int n)
{
    for (int r = 0; r < n; r++)
        for (int col = 0; col < n; col++) {
            int sum = 0;
            for (int k = 0; k < n; k++)
                sum += a[r * n + k] * b[k * n + col];
            c[r * n + col] = sum;
        }
}

bool verify(const int *gpu, const int *cpu, int len)
{
    for (int i = 0; i < len; i++)
        if (gpu[i] != cpu[i])
            return false;
    return true;
}

void printMatrix(const char *name, const int *m, int n)
{
    printf("%s =\n", name);
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++)
            printf("%5d ", m[r * n + c]);
        printf("\n");
    }
}

void run(int n, bool show, int device, int blocks, cudaStream_t sA, cudaStream_t sB)
{
    size_t bytes = (size_t)n * n * sizeof(int);
    int *a, *b, *c, *cHost;
    long long *sum;
    CHECK(cudaMallocManaged(&a, bytes));
    CHECK(cudaMallocManaged(&b, bytes));
    CHECK(cudaMallocManaged(&c, bytes));
    CHECK(cudaMallocManaged(&sum, sizeof(long long)));
    cHost = (int *)malloc(bytes);
    *sum = 0;

    prefetch(a, bytes, device, sA);
    prefetch(b, bytes, device, sB);
    prefetch(c, bytes, device);
    prefetch(sum, sizeof(long long), device);

    fillA<<<blocks, THREADS, 0, sA>>>(a, n);
    fillB<<<blocks, THREADS, 0, sB>>>(b, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    matMul<<<blocks, THREADS>>>(a, b, c, n);
    CHECK(cudaEventRecord(stop));
    reduceSum<<<blocks, THREADS>>>(c, sum, n * n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    float gpuMs = 0;
    CHECK(cudaEventElapsedTime(&gpuMs, start, stop));

    prefetch(a, bytes, cudaCpuDeviceId);
    prefetch(b, bytes, cudaCpuDeviceId);
    prefetch(c, bytes, cudaCpuDeviceId);
    prefetch(sum, sizeof(long long), cudaCpuDeviceId);
    CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    matMulHost(a, b, cHost, n);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    long long cpuSum = 0;
    for (int i = 0; i < n * n; i++)
        cpuSum += cHost[i];

    printf("\nN = %d\n", n);
    if (show) {
        printMatrix("A", a, n);
        printMatrix("B", b, n);
        printMatrix("C", c, n);
    }
    printf("Sum (C) = %lld\n", *sum);
    printf("C[0][0] = %d\n", c[0]);
    printf("C[N-1][N-1] = %d\n", c[n * n - 1]);
    printf("Time: CPU %.3f ms, GPU %.3f ms\n", cpuMs, gpuMs);
    printf("Verify: %s\n", verify(c, cHost, n * n) && *sum == cpuSum ? "PASSED" : "FAILED");

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(a));
    CHECK(cudaFree(b));
    CHECK(cudaFree(c));
    CHECK(cudaFree(sum));
    free(cHost);
}

int main()
{
    int device;
    CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device));
    int blocks = prop.warpSize * prop.multiProcessorCount;
    printf("Hardware name: %s\n", prop.name);

    cudaStream_t sA, sB;
    CHECK(cudaStreamCreate(&sA));
    CHECK(cudaStreamCreate(&sB));

    run(4, true, device, blocks, sA, sB);
    run(1 << 8, false, device, blocks, sA, sB);

    CHECK(cudaStreamDestroy(sA));
    CHECK(cudaStreamDestroy(sB));
    return 0;
}
