#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#define CHECK(call)                                                           \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err),       \
                   __FILE__, __LINE__);                                       \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// Multiple of warp size (32); 256 gives good occupancy on most GPUs
const int THREADS = 256;

enum Op { ADD = 0, SUB = 1, MUL = 2 };

// ---------------- device functions ----------------
__device__ int addDevice(int x, int y) { return x + y; }
__device__ int subDevice(int x, int y) { return x - y; }
__device__ int mulDevice(int x, int y) { return x * y; }

// ---------------- kernels ----------------
__global__ void fillKernel(int *arr, int n, unsigned seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        unsigned h = (unsigned)i * 2654435761u + seed;
        h ^= h >> 13;
        h *= 0x5bd1e995u;
        h ^= h >> 15;
        arr[i] = (int)(h % 11) - 5; // [-5, 5]
    }
}

__global__ void opKernel(const int *a, const int *b, int *c, int n, Op op)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        if (op == ADD)      c[i] = addDevice(a[i], b[i]);
        else if (op == SUB) c[i] = subDevice(a[i], b[i]);
        else                c[i] = mulDevice(a[i], b[i]);
    }
}

// Shared-memory tree reduction; blockDim.x must be a power of two
__global__ void reduceKernel(const int *in, long long *out, int n)
{
    __shared__ long long cache[THREADS];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    long long sum = 0;
    for (int i = idx; i < n; i += stride)
        sum += in[i];
    cache[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            cache[tid] += cache[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd((unsigned long long *)out, (unsigned long long)cache[0]);
}

// ---------------- host functions ----------------
void opHost(const int *a, const int *b, int *c, int n, Op op)
{
    for (int i = 0; i < n; i++) {
        if (op == ADD)      c[i] = a[i] + b[i];
        else if (op == SUB) c[i] = a[i] - b[i];
        else                c[i] = a[i] * b[i];
    }
}

long long sumHost(const int *c, int n)
{
    long long s = 0;
    for (int i = 0; i < n; i++) s += c[i];
    return s;
}

bool verify(const int *gpu, const int *cpu, int n, long long gpuSum, long long cpuSum)
{
    for (int i = 0; i < n; i++)
        if (gpu[i] != cpu[i]) return false;
    return gpuSum == cpuSum;
}

void printArray(const char *label, const int *arr, int n)
{
    printf("%s", label);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    printf("\n");
}

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

void run(int n, bool printArrays, int device, int blocks, cudaStream_t *streams)
{
    size_t bytes = n * sizeof(int);
    int *a, *b, *cGpu[3], *cCpu[3];
    long long *sums;

    CHECK(cudaMallocManaged(&a, bytes));
    CHECK(cudaMallocManaged(&b, bytes));
    for (int k = 0; k < 3; k++) {
        CHECK(cudaMallocManaged(&cGpu[k], bytes));
        cCpu[k] = (int *)malloc(bytes);
    }
    CHECK(cudaMallocManaged(&sums, 3 * sizeof(long long)));

    // async migration to GPU before kernels touch the data
    prefetch(a, bytes, device, streams[0]);
    prefetch(b, bytes, device, streams[1]);
    for (int k = 0; k < 3; k++) prefetch(cGpu[k], bytes, device, streams[k]);

    // fill A and B on independent streams
    fillKernel<<<blocks, THREADS, 0, streams[0]>>>(a, n, 42u);
    fillKernel<<<blocks, THREADS, 0, streams[1]>>>(b, n, 1337u);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    const char *names[3] = {"addition", "subtraction", "multiplication"};

    // ---------------- GPU ----------------
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    for (int k = 0; k < 3; k++) sums[k] = 0;
    prefetch(sums, 3 * sizeof(long long), device, 0);
    CHECK(cudaDeviceSynchronize());

    for (int k = 0; k < 3; k++) {
        if (k == MUL) CHECK(cudaEventRecord(start, streams[k]));
        opKernel<<<blocks, THREADS, 0, streams[k]>>>(a, b, cGpu[k], n, (Op)k);
        if (k == MUL) CHECK(cudaEventRecord(stop, streams[k]));
        reduceKernel<<<blocks, THREADS, 0, streams[k]>>>(cGpu[k], &sums[k], n);
    }
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    float gpuMs = 0;
    CHECK(cudaEventElapsedTime(&gpuMs, start, stop));

    // async migration back to CPU before host reads
    prefetch(a, bytes, cudaCpuDeviceId, 0);
    prefetch(b, bytes, cudaCpuDeviceId, 0);
    for (int k = 0; k < 3; k++) prefetch(cGpu[k], bytes, cudaCpuDeviceId, 0);
    prefetch(sums, 3 * sizeof(long long), cudaCpuDeviceId, 0);
    CHECK(cudaDeviceSynchronize());

    // ---------------- CPU ----------------
    long long cpuSums[3];
    double cpuMs = 0;
    for (int k = 0; k < 3; k++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        opHost(a, b, cCpu[k], n, (Op)k);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (k == MUL) cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        cpuSums[k] = sumHost(cCpu[k], n);
    }

    if (printArrays) {
        printf("Generated arrays when N = %d:\n", n);
        printArray("A = ", a, n);
        printArray("B = ", b, n);

        printf("\nCPU results:\n");
        for (int k = 0; k < 3; k++) {
            printf("C (%s) = ", names[k]);
            printArray("", cCpu[k], n);
        }
        for (int k = 0; k < 3; k++) printf("Sum (%s) = %lld\n", names[k], cpuSums[k]);

        cudaDeviceProp prop;
        CHECK(cudaGetDeviceProperties(&prop, device));
        printf("\nGPU results:\n");
        printf("Hardware name: %s\n", prop.name);
        for (int k = 0; k < 3; k++) {
            printf("C (%s) = ", names[k]);
            printArray("", cGpu[k], n);
        }
        for (int k = 0; k < 3; k++) printf("Sum (%s) = %lld\n", names[k], sums[k]);
    } else {
        printf("\nCPU time to complete multiplication when N = %d: %.3f ms\n", n, cpuMs);
        printf("GPU time to complete multiplication when N = %d: %.3f ms\n", n, gpuMs);
    }

    for (int k = 0; k < 3; k++) {
        bool ok = verify(cGpu[k], cCpu[k], n, sums[k], cpuSums[k]);
        printf("Verify %s (N = %d): %s\n", names[k], n, ok ? "PASSED" : "FAILED");
    }

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(a));
    CHECK(cudaFree(b));
    for (int k = 0; k < 3; k++) {
        CHECK(cudaFree(cGpu[k]));
        free(cCpu[k]);
    }
    CHECK(cudaFree(sums));
}

int main()
{
    int device;
    CHECK(cudaGetDevice(&device));

    int numSMs, warp;
    CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));
    CHECK(cudaDeviceGetAttribute(&warp, cudaDevAttrWarpSize, device));
    int blocks = warp * numSMs;
    printf("SMs = %d, warp size = %d, blocks = %d, threads = %d\n\n", numSMs, warp, blocks, THREADS);

    cudaStream_t streams[3];
    for (int k = 0; k < 3; k++) CHECK(cudaStreamCreate(&streams[k]));

    run(10, true, device, blocks, streams);
    run(2 << 20, false, device, blocks, streams);

    for (int k = 0; k < 3; k++) CHECK(cudaStreamDestroy(streams[k]));
    return 0;
}
