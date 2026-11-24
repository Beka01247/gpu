// Universal starter for the CUDA midterm tasks.
// Compiles and runs as is (c = a + b, sum of c). Lines marked CHANGE are task-specific;
// everything else is the same in every task.
// Matrices: allocate N * N elements, loop i < n * n, row = i / n, col = i % n.

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

// multiple of warp size, power of two for the reduction
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

// ============================ device functions ============================
// CHANGE: one __device__ function per operation (add, sub, mul, poly, dot row x col, ...)
__device__ int opDevice(int x, int y) { return x + y; }

// ================================ kernels =================================
// CHANGE: fill formulas from the task (or your own rule from the index)
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

// CHANGE: one kernel per operation; keep the grid-stride loop, change only the body
__global__ void opKernel(const int *a, const int *b, int *c, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        c[i] = opDevice(a[i], b[i]);
}

// never changes: sum of n ints; *result must be 0 before the launch
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

// only if the task asks for max/min (delete otherwise): *result = INT_MIN before the launch
// (for min: INT_MAX start, min() instead of max(), atomicMin)
__global__ void reduceMax(const int *in, int *result, int n)
{
    __shared__ int cache[THREADS];
    int tid = threadIdx.x;

    int m = INT_MIN;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        m = max(m, in[i]);
    cache[tid] = m;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            cache[tid] = max(cache[tid], cache[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        atomicMax(result, cache[0]);
}

// ============================= host functions =============================
// CHANGE: the same computation as opKernel, on the CPU
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

// ================================== main ==================================
int main()
{
    const int N = 2 << 20;                       // CHANGE: size from the task
    size_t bytes = (size_t)N * sizeof(int);      // CHANGE: matrices -> (size_t)N * N * sizeof(int)

    // 1. device info -> blocks = warp size * SMs
    int device;
    CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device));
    int blocks = prop.warpSize * prop.multiProcessorCount;
    printf("Hardware name: %s\n", prop.name);

    // 2. memory: one managed array per input/output, one long long per sum
    int *a, *b, *c, *cHost;                      // CHANGE: add arrays the task needs
    long long *sum;
    CHECK(cudaMallocManaged(&a, bytes));
    CHECK(cudaMallocManaged(&b, bytes));
    CHECK(cudaMallocManaged(&c, bytes));
    CHECK(cudaMallocManaged(&sum, sizeof(long long)));
    cHost = (int *)malloc(bytes);
    *sum = 0;                                    // set results on the CPU before any kernel

    // 3. streams: inputs are filled on independent streams
    cudaStream_t s1, s2;
    CHECK(cudaStreamCreate(&s1));
    CHECK(cudaStreamCreate(&s2));

    // 4. async migration to the GPU
    prefetch(a, bytes, device, s1);
    prefetch(b, bytes, device, s2);
    prefetch(c, bytes, device);
    prefetch(sum, sizeof(long long), device);

    // 5. fill inputs on the GPU
    fillA<<<blocks, THREADS, 0, s1>>>(a, N);
    fillB<<<blocks, THREADS, 0, s2>>>(b, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // 6. compute + reduce; the timed kernel sits between the two events
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    opKernel<<<blocks, THREADS>>>(a, b, c, N);   // CHANGE: the kernel(s) of the task
    CHECK(cudaEventRecord(stop));
    reduceSum<<<blocks, THREADS>>>(c, sum, N);   // CHANGE: one reduceSum per required sum
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    float gpuMs = 0;
    CHECK(cudaEventElapsedTime(&gpuMs, start, stop));

    // 7. async migration back to the CPU, then wait before reading anything
    prefetch(a, bytes, cudaCpuDeviceId);
    prefetch(b, bytes, cudaCpuDeviceId);
    prefetch(c, bytes, cudaCpuDeviceId);
    prefetch(sum, sizeof(long long), cudaCpuDeviceId);
    CHECK(cudaDeviceSynchronize());

    // 8. host version, timed with chrono
    auto t0 = std::chrono::high_resolution_clock::now();
    opHost(a, b, cHost, N);                      // CHANGE: host version of the timed operation
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    long long cpuSum = sumHost(cHost, N);

    // 9. output: CHANGE labels and values to the format from the task
    printArray("A (first 10)", a, 10);
    printArray("B (first 10)", b, 10);
    printf("CPU results:\nSum (addition) = %lld\n", cpuSum);
    printf("GPU results:\nSum (addition) = %lld\n", *sum);
    printf("Time to complete: CPU %.3f ms, GPU %.3f ms\n", cpuMs, gpuMs);
    bool ok = verify(c, cHost, N) && *sum == cpuSum;
    printf("Verify: %s\n", ok ? "PASSED" : "FAILED");

    // 10. clear memory: every cudaMalloc* -> cudaFree, every malloc -> free
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
