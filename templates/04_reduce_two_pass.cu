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

// no atomics: every block writes its partial sum, the host adds them
__global__ void reducePartial(const int *in, long long *partial, int n)
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
        partial[blockIdx.x] = cache[0];
}

int main()
{
    const int N = 1 << 20;
    int blocks = numBlocks();
    int *a;
    long long *partial;
    CHECK(cudaMallocManaged(&a, N * sizeof(int)));
    CHECK(cudaMallocManaged(&partial, blocks * sizeof(long long)));
    long long expected = 0;
    for (int i = 0; i < N; i++) {
        a[i] = i % 7 - 3;
        expected += a[i];
    }

    reducePartial<<<blocks, THREADS>>>(a, partial, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    long long total = 0;
    for (int k = 0; k < blocks; k++)
        total += partial[k];
    printf("Sum = %lld (expected %lld) -> %s\n", total, expected, total == expected ? "PASSED" : "FAILED");
    CHECK(cudaFree(a));
    CHECK(cudaFree(partial));
    return 0;
}
