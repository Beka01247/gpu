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

// multiple of warp size, power of two
const int THREADS = 256;

__global__ void fillMatrix(int *m, int n, int seed)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * n; i += blockDim.x * gridDim.x)
        m[i] = (i * 7 + seed) % 10;
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

bool verify(const int *gpu, const int *cpu, int n)
{
    for (int i = 0; i < n * n; i++)
        if (gpu[i] != cpu[i]) {
            printf("Mismatch at (%d, %d): GPU %d, CPU %d\n", i / n, i % n, gpu[i], cpu[i]);
            return false;
        }
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

void run(int n, bool printMatrices, int device, int blocks, cudaStream_t sA, cudaStream_t sB)
{
    size_t bytes = (size_t)n * n * sizeof(int);
    int *a, *b, *c, *cHost;
    CHECK(cudaMallocManaged(&a, bytes));
    CHECK(cudaMallocManaged(&b, bytes));
    CHECK(cudaMallocManaged(&c, bytes));
    cHost = (int *)malloc(bytes);

    prefetch(a, bytes, device, sA);
    prefetch(b, bytes, device, sB);
    prefetch(c, bytes, device);

    fillMatrix<<<blocks, THREADS, 0, sA>>>(a, n, 1);
    fillMatrix<<<blocks, THREADS, 0, sB>>>(b, n, 3);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    matMulKernel<<<blocks, THREADS>>>(a, b, c, n);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaGetLastError());
    CHECK(cudaEventSynchronize(stop));
    float gpuMs = 0;
    CHECK(cudaEventElapsedTime(&gpuMs, start, stop));

    prefetch(a, bytes, cudaCpuDeviceId);
    prefetch(b, bytes, cudaCpuDeviceId);
    prefetch(c, bytes, cudaCpuDeviceId);
    CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    matMulHost(a, b, cHost, n);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("\nN = %d\n", n);
    if (printMatrices) {
        printMatrix("A", a, n);
        printMatrix("B", b, n);
        printMatrix("C (GPU)", c, n);
        printMatrix("C (CPU)", cHost, n);
    }
    printf("Time to complete (kernel): %.3f ms\n", gpuMs);
    printf("Time to complete (host):   %.3f ms\n", cpuMs);
    printf("Verify: %s\n", verify(c, cHost, n) ? "PASSED" : "FAILED");

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(a));
    CHECK(cudaFree(b));
    CHECK(cudaFree(c));
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
    printf("SMs = %d, warp size = %d, blocks = %d, threads = %d\n",
           prop.multiProcessorCount, prop.warpSize, blocks, THREADS);

    cudaStream_t sA, sB;
    CHECK(cudaStreamCreate(&sA));
    CHECK(cudaStreamCreate(&sB));

    run(4, true, device, blocks, sA, sB);
    run(1 << 6, false, device, blocks, sA, sB);
    run(1 << 10, false, device, blocks, sA, sB);

    CHECK(cudaStreamDestroy(sA));
    CHECK(cudaStreamDestroy(sB));
    return 0;
}
