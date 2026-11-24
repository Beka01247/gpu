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

// ---------- device functions ----------
__device__ int opDevice(int x, int y) { return x + y; }

// ---------- kernels ----------
__global__ void fillA(int *a, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        a[i] = (i % 11) - 5;
}

__global__ void fillB(int *b, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        b[i] = ((i * 7 + 3) % 11) - 5;
}

__global__ void opKernel(const int *a, const int *b, int *c, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        c[i] = opDevice(a[i], b[i]);
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

// ---------- host functions ----------
void opHost(const int *a, const int *b, int *c, int n)
{
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

long long sumHost(const int *c, int n)
{
    long long s = 0;
    for (int i = 0; i < n; i++)
        s += c[i];
    return s;
}

bool verify(const int *gpu, const int *cpu, int n)
{
    for (int i = 0; i < n; i++)
        if (gpu[i] != cpu[i]) {
            printf("Mismatch at %d: GPU %d, CPU %d\n", i, gpu[i], cpu[i]);
            return false;
        }
    return true;
}

void printArray(const char *name, const int *x, int n)
{
    printf("%s = ", name);
    for (int i = 0; i < n; i++)
        printf("%d ", x[i]);
    printf("\n");
}

// ---------- main ----------
int main()
{
    const int N = 2 << 20;
    size_t bytes = N * sizeof(int);

    // 1. device info -> blocks
    int device;
    CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device));
    int numSMs = prop.multiProcessorCount;
    int blocks = prop.warpSize * numSMs;
    printf("Hardware name: %s\n", prop.name);
    printf("SMs = %d, blocks = %d, threads = %d\n", numSMs, blocks, THREADS);

    // 2. memory
    int *a, *b, *c, *cHost;
    long long *sum;
    CHECK(cudaMallocManaged(&a, bytes));
    CHECK(cudaMallocManaged(&b, bytes));
    CHECK(cudaMallocManaged(&c, bytes));
    CHECK(cudaMallocManaged(&sum, sizeof(long long)));
    cHost = (int *)malloc(bytes);
    *sum = 0;

    // 3. streams
    cudaStream_t s1, s2;
    CHECK(cudaStreamCreate(&s1));
    CHECK(cudaStreamCreate(&s2));

    // 4. async migration to GPU
    prefetch(a, bytes, device, s1);
    prefetch(b, bytes, device, s2);
    prefetch(c, bytes, device);
    prefetch(sum, sizeof(long long), device);

    // 5. fill on independent streams
    fillA<<<blocks, THREADS, 0, s1>>>(a, N);
    fillB<<<blocks, THREADS, 0, s2>>>(b, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // 6. compute + reduce, timed with events
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    opKernel<<<blocks, THREADS>>>(a, b, c, N);
    CHECK(cudaEventRecord(stop));
    reduceSum<<<blocks, THREADS>>>(c, sum, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    float gpuMs = 0;
    CHECK(cudaEventElapsedTime(&gpuMs, start, stop));

    // 7. async migration back to CPU
    prefetch(a, bytes, cudaCpuDeviceId);
    prefetch(b, bytes, cudaCpuDeviceId);
    prefetch(c, bytes, cudaCpuDeviceId);
    prefetch(sum, sizeof(long long), cudaCpuDeviceId);
    CHECK(cudaDeviceSynchronize());

    // 8. CPU version, timed with chrono
    auto t0 = std::chrono::high_resolution_clock::now();
    opHost(a, b, cHost, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    long long cpuSum = sumHost(cHost, N);

    // 9. output
    printArray("A (first 10)", a, 10);
    printArray("B (first 10)", b, 10);
    printf("CPU results:\nSum (addition) = %lld\n", cpuSum);
    printf("GPU results:\nSum (addition) = %lld\n", *sum);
    printf("Time CPU = %.3f ms, GPU = %.3f ms\n", cpuMs, gpuMs);
    bool ok = verify(c, cHost, N) && (*sum == cpuSum);
    printf("Verify: %s\n", ok ? "PASSED" : "FAILED");

    // 10. clear memory
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaStreamDestroy(s1));
    CHECK(cudaStreamDestroy(s2));
    CHECK(cudaFree(a));
    CHECK(cudaFree(b));
    CHECK(cudaFree(c));
    CHECK(cudaFree(sum));
    free(cHost);
    return 0;
}
