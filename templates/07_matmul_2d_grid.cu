#include <cstdio>
#include <cstdlib>
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

// alternative launch: 2D blocks and a 2D grid-stride loop
__device__ int dotRowCol(const int *a, const int *b, int n, int row, int col)
{
    int sum = 0;
    for (int k = 0; k < n; k++)
        sum += a[row * n + k] * b[k * n + col];
    return sum;
}

__global__ void matMul2D(const int *a, const int *b, int *c, int n)
{
    for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < n; row += blockDim.y * gridDim.y)
        for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < n; col += blockDim.x * gridDim.x)
            c[row * n + col] = dotRowCol(a, b, n, row, col);
}

int main()
{
    const int N = 128;
    size_t bytes = (size_t)N * N * sizeof(int);
    int *a, *b, *c;
    CHECK(cudaMallocManaged(&a, bytes));
    CHECK(cudaMallocManaged(&b, bytes));
    CHECK(cudaMallocManaged(&c, bytes));
    for (int i = 0; i < N * N; i++) {
        a[i] = i % 10;
        b[i] = (i * 3) % 7;
    }

    dim3 threads(16, 16);
    dim3 blocks(8, 8);
    matMul2D<<<blocks, threads>>>(a, b, c, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    bool ok = true;
    for (int r = 0; r < N; r++)
        for (int col = 0; col < N; col++) {
            int v = 0;
            for (int k = 0; k < N; k++)
                v += a[r * N + k] * b[k * N + col];
            ok = ok && c[r * N + col] == v;
        }
    printf("Verify: %s\n", ok ? "PASSED" : "FAILED");
    CHECK(cudaFree(a));
    CHECK(cudaFree(b));
    CHECK(cudaFree(c));
    return 0;
}
