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

// explicit memory: pinned host buffers + device buffers + cudaMemcpyAsync
__global__ void addKernel(const int *a, const int *b, int *c, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        c[i] = a[i] + b[i];
}

int main()
{
    const int N = 1 << 20;
    size_t bytes = N * sizeof(int);

    int *hA, *hB, *hC;
    CHECK(cudaMallocHost(&hA, bytes));
    CHECK(cudaMallocHost(&hB, bytes));
    CHECK(cudaMallocHost(&hC, bytes));
    for (int i = 0; i < N; i++) {
        hA[i] = i % 10;
        hB[i] = i % 7;
    }

    int *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytes));
    CHECK(cudaMalloc(&dB, bytes));
    CHECK(cudaMalloc(&dC, bytes));

    cudaStream_t s;
    CHECK(cudaStreamCreate(&s));
    CHECK(cudaMemcpyAsync(dA, hA, bytes, cudaMemcpyHostToDevice, s));
    CHECK(cudaMemcpyAsync(dB, hB, bytes, cudaMemcpyHostToDevice, s));
    addKernel<<<numBlocks(), THREADS, 0, s>>>(dA, dB, dC, N);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpyAsync(hC, dC, bytes, cudaMemcpyDeviceToHost, s));
    CHECK(cudaStreamSynchronize(s));

    bool ok = true;
    for (int i = 0; i < N; i++)
        ok = ok && hC[i] == hA[i] + hB[i];
    printf("Verify: %s\n", ok ? "PASSED" : "FAILED");

    CHECK(cudaStreamDestroy(s));
    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));
    CHECK(cudaFreeHost(hA));
    CHECK(cudaFreeHost(hB));
    CHECK(cudaFreeHost(hC));
    return 0;
}
