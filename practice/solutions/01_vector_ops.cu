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

__device__ int addDevice(int x, int y) { return x + y; }
__device__ int subDevice(int x, int y) { return x - y; }
__device__ int mulDevice(int x, int y) { return x * y; }

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

__global__ void addKernel(const int *a, const int *b, int *c, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        c[i] = addDevice(a[i], b[i]);
}

__global__ void subKernel(const int *a, const int *b, int *c, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        c[i] = subDevice(a[i], b[i]);
}

__global__ void mulKernel(const int *a, const int *b, int *c, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        c[i] = mulDevice(a[i], b[i]);
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

void addHost(const int *a, const int *b, int *c, int n) { for (int i = 0; i < n; i++) c[i] = a[i] + b[i]; }
void subHost(const int *a, const int *b, int *c, int n) { for (int i = 0; i < n; i++) c[i] = a[i] - b[i]; }
void mulHost(const int *a, const int *b, int *c, int n) { for (int i = 0; i < n; i++) c[i] = a[i] * b[i]; }

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
        if (gpu[i] != cpu[i])
            return false;
    return true;
}

void printArray(const char *name, const int *x, int n)
{
    printf("%s = ", name);
    for (int i = 0; i < n; i++)
        printf("%d ", x[i]);
    printf("\n");
}

int main()
{
    const int N = 2 << 20;
    size_t bytes = N * sizeof(int);
    const char *names[3] = {"addition", "subtraction", "multiplication"};

    int device;
    CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device));
    int blocks = prop.warpSize * prop.multiProcessorCount;
    printf("Hardware name: %s\n", prop.name);

    int *a, *b, *c[3], *cHost[3];
    long long *sums;
    CHECK(cudaMallocManaged(&a, bytes));
    CHECK(cudaMallocManaged(&b, bytes));
    for (int k = 0; k < 3; k++) {
        CHECK(cudaMallocManaged(&c[k], bytes));
        cHost[k] = (int *)malloc(bytes);
    }
    CHECK(cudaMallocManaged(&sums, 3 * sizeof(long long)));
    for (int k = 0; k < 3; k++)
        sums[k] = 0;

    cudaStream_t s[3];
    for (int k = 0; k < 3; k++)
        CHECK(cudaStreamCreate(&s[k]));

    CHECK(cudaMemPrefetchAsync(a, bytes, device, s[0]));
    CHECK(cudaMemPrefetchAsync(b, bytes, device, s[1]));
    for (int k = 0; k < 3; k++)
        CHECK(cudaMemPrefetchAsync(c[k], bytes, device, s[k]));
    CHECK(cudaMemPrefetchAsync(sums, 3 * sizeof(long long), device));

    fillA<<<blocks, THREADS, 0, s[0]>>>(a, N);
    fillB<<<blocks, THREADS, 0, s[1]>>>(b, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    addKernel<<<blocks, THREADS, 0, s[0]>>>(a, b, c[0], N);
    subKernel<<<blocks, THREADS, 0, s[1]>>>(a, b, c[1], N);
    CHECK(cudaEventRecord(start, s[2]));
    mulKernel<<<blocks, THREADS, 0, s[2]>>>(a, b, c[2], N);
    CHECK(cudaEventRecord(stop, s[2]));
    for (int k = 0; k < 3; k++)
        reduceSum<<<blocks, THREADS, 0, s[k]>>>(c[k], &sums[k], N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    float gpuMs = 0;
    CHECK(cudaEventElapsedTime(&gpuMs, start, stop));

    CHECK(cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId));
    CHECK(cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId));
    for (int k = 0; k < 3; k++)
        CHECK(cudaMemPrefetchAsync(c[k], bytes, cudaCpuDeviceId));
    CHECK(cudaMemPrefetchAsync(sums, 3 * sizeof(long long), cudaCpuDeviceId));
    CHECK(cudaDeviceSynchronize());

    addHost(a, b, cHost[0], N);
    subHost(a, b, cHost[1], N);
    auto t0 = std::chrono::high_resolution_clock::now();
    mulHost(a, b, cHost[2], N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("Generated arrays when N = 10:\n");
    printArray("A", a, 10);
    printArray("B", b, 10);
    printf("GPU results (first 10):\n");
    for (int k = 0; k < 3; k++) {
        printf("C (%s)", names[k]);
        printArray("", c[k], 10);
    }

    bool ok = true;
    for (int k = 0; k < 3; k++) {
        long long cpuSum = sumHost(cHost[k], N);
        ok = ok && verify(c[k], cHost[k], N) && sums[k] == cpuSum;
        printf("Sum (%s) = %lld\n", names[k], sums[k]);
    }
    printf("Time to complete multiplication when N = 2<<20: CPU %.3f ms, GPU %.3f ms\n", cpuMs, gpuMs);
    printf("Verify: %s\n", ok ? "PASSED" : "FAILED");

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    for (int k = 0; k < 3; k++) {
        CHECK(cudaStreamDestroy(s[k]));
        CHECK(cudaFree(c[k]));
        free(cHost[k]);
    }
    CHECK(cudaFree(a));
    CHECK(cudaFree(b));
    CHECK(cudaFree(sums));
    return 0;
}
