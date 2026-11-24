#include <cstdio>
#include <cstdlib>
#include <cmath>
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

int main()
{
    size_t freeBytes, totalBytes;
    CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    printf("GPU memory: free %zu MB of %zu MB\n", freeBytes >> 20, totalBytes >> 20);

    // three int matrices of N x N have to fit into free memory
    int n = (int)sqrt((double)freeBytes / (3 * sizeof(int)));
    printf("Largest N for C = A x B with int matrices: %d\n", n);

    int *probe;
    size_t bytes = (size_t)n * n * sizeof(int);
    bool ok = cudaMalloc(&probe, bytes) == cudaSuccess;
    if (ok)
        CHECK(cudaFree(probe));
    printf("Allocating one %d x %d matrix: %s\n", n, n, ok ? "OK" : "failed");
    return 0;
}
