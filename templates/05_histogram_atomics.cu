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

const int BINS = 11;  // values -5..5

__global__ void histogram(const int *a, int *hist, int n)
{
    __shared__ int local[BINS];
    for (int j = threadIdx.x; j < BINS; j += blockDim.x)
        local[j] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        atomicAdd(&local[a[i] + 5], 1);
    __syncthreads();
    for (int j = threadIdx.x; j < BINS; j += blockDim.x)
        atomicAdd(&hist[j], local[j]);
}

__global__ void countPositive(const int *a, int *count, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        if (a[i] > 0)
            atomicAdd(count, 1);
}

int main()
{
    const int N = 1 << 20;
    int *a, *hist, *count;
    CHECK(cudaMallocManaged(&a, N * sizeof(int)));
    CHECK(cudaMallocManaged(&hist, BINS * sizeof(int)));
    CHECK(cudaMallocManaged(&count, sizeof(int)));
    int expHist[BINS] = {0}, expCount = 0;
    for (int i = 0; i < N; i++) {
        a[i] = (i * 31 + 7) % 11 - 5;
        expHist[a[i] + 5]++;
        expCount += a[i] > 0;
    }
    for (int j = 0; j < BINS; j++)
        hist[j] = 0;
    *count = 0;

    int blocks = numBlocks();
    histogram<<<blocks, THREADS>>>(a, hist, N);
    countPositive<<<blocks, THREADS>>>(a, count, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    bool ok = *count == expCount;
    for (int j = 0; j < BINS; j++) {
        printf("Bin %d = %d\n", j - 5, hist[j]);
        ok = ok && hist[j] == expHist[j];
    }
    printf("Count (a > 0) = %d -> %s\n", *count, ok ? "PASSED" : "FAILED");
    CHECK(cudaFree(a));
    CHECK(cudaFree(hist));
    CHECK(cudaFree(count));
    return 0;
}
