#include <cstdio>
#include <cstdlib>
#include <climits>
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

int numBlocks()
{
    int device, sms, warp;
    CHECK(cudaGetDevice(&device));
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    CHECK(cudaDeviceGetAttribute(&warp, cudaDevAttrWarpSize, device));
    return warp * sms;
}

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

__global__ void reduceMin(const int *in, int *result, int n)
{
    __shared__ int cache[THREADS];
    int tid = threadIdx.x;
    int m = INT_MAX;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        m = min(m, in[i]);
    cache[tid] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            cache[tid] = min(cache[tid], cache[tid + s]);
        __syncthreads();
    }
    if (tid == 0)
        atomicMin(result, cache[0]);
}

int main()
{
    const int N = 1 << 20;
    int *a, *mx, *mn;
    CHECK(cudaMallocManaged(&a, N * sizeof(int)));
    CHECK(cudaMallocManaged(&mx, sizeof(int)));
    CHECK(cudaMallocManaged(&mn, sizeof(int)));
    for (int i = 0; i < N; i++)
        a[i] = (i * 37) % 1001 - 500;
    *mx = INT_MIN;
    *mn = INT_MAX;

    int blocks = numBlocks();
    reduceMax<<<blocks, THREADS>>>(a, mx, N);
    reduceMin<<<blocks, THREADS>>>(a, mn, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    printf("Max = %d, Min = %d -> %s\n", *mx, *mn, *mx == 500 && *mn == -500 ? "PASSED" : "FAILED");
    CHECK(cudaFree(a));
    CHECK(cudaFree(mx));
    CHECK(cudaFree(mn));
    return 0;
}
