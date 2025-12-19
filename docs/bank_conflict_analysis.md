# Deep dive with Bank Conflict

## Motivation

This project studies the performance characteristics of CUDA GEMM implementations, with a particular focus on shared memory bank conflicts.

While many GEMM optimizations are often presented as implementation tricks, this repository aims to derive these optimizations from first principles, starting from naive CUDA kernels and analyzing how hardware-level constraints influence performance.

## Baseline: Naive CUDA GEMM

We begin with a naive CUDA GEMM implementation:

```cuda
__global__ void gemm_naive(float *A, float *B, float *C,
                           int M, int N, int K) {
    //global thread index
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    //inner product calculation
    float tmp = 0.0f;
    if (row < M && col < N) {
        for (int k = 0; k < K; ++k) {
            tmp += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = tmp;
    }
}
```

### Memory Access Observation

- Threads within a warp are contiguous in `threadIdx.x`
- Access to `B[k * N + col]` is naturally coalesced
- Access to `A[row * K + k]` is broadcast across the warp


Observation: This kernel is severely bandwidth-limited. Every floating-point operation (FLOP) requires two global memory fetches (assuming no L1/L2 cache hits), resulting in very low Arithmetic Intensity.
## Shared Memory Tiling

To reduce Global Memory traffic, we introduce Shared Memory Tiling. This explicitly manages a "programmable cache":

```cuda
#define TILE_SIZE 64

__global__ void gemm_tiled(float *A, float *B, float *C,
                           int M, int N, int K) {
    //Shared memory allocation
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    //global thread index and block thread index 
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float tmp = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    //OUTER LOOP (Loading Tiles)
    for (int t = 0; t < num_tiles; ++t) {
        // Collaborative loading from Global to Shared Memory
        As[ty][tx] = (row < M && t * TILE_SIZE + tx < K)
                     ? A[row * K + t * TILE_SIZE + tx] : 0.0f;

        Bs[ty][tx] = (t * TILE_SIZE + ty < K && col < N)
                     ? B[(t * TILE_SIZE + ty) * N + col] : 0.0f;

        __syncthreads();
        //INNER LOOP (Compute on Shared Memory)
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            // Potential Bank Conflict Zone
            tmp += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = tmp;
}
```

This reduces global memory traffic and exposes shared memory behavior.

## Bank Conflict Analysis

### Bank Mapping Rule

On NVIDIA GPUs (Shared Memory):

- Banks: Memory is divided into 32 parallel banks.
- Bandwidth: Each bank can serve 32 bits (4 bytes) per clock cycle.
- Bank index is computed as:

```
bank_id = (address_in_bytes / 4) % 32
```

Thus, a float (4 bytes) maps exactly to one bank.

### The Cost of Conflicts
- No Conflict: If all 32 threads in a Warp access distinct banks (or the same address via broadcast), the hardware serves all requests in 1 transaction.
- N-way Conflict: If $N$ threads within a Warp access different addresses that map to the same bank, the hardware serializes the request into $N$ separate transactions.
- Performance Hit: An 32-way conflict reduces shared memory effective bandwidth by 1/32.
### Row-wise Access Pattern (Conflict-Free)

In the inner loop:

```cuda
As[ty][k]   // k varies
Bs[k][tx]   // tx varies across warp
```

`Bs[k][tx]` is accessed with contiguous `tx` → This maps to consecutive banks → No bank conflict occurs.

### Column-wise Access Pattern (Conflict-Prone)

If instead the layout were:

```cuda
Bs[tx][k]
```

Then:

- Threads in a warp access different rows but same column
- Bank index depends on row stride
- When `TILE_SIZE % 32 == 0`, all threads map to the same bank
- This causes 32-way bank conflicts.

### Effect of TILE_SIZE

| TILE_SIZE | Row stride (floats) | Bank behavior |
|-----------|---------------------|---------------|
| 16        | 16                  | Banks wrap every 2 rows |
| 32        | 32                  | Each row maps to same bank set |
| 64        | 64                  | Full bank reuse per row |

Bank conflicts are determined by:

```
(row_stride_in_floats % 32)
```

### Data Type Effects

| Type | Bytes | Elements per bank |
|------|-------|-------------------|
| fp32 | 4     | 1                 |
| fp16 | 2     | 2                 |
| fp8  | 1     | 4                 |

**Lower-precision data types increase the likelihood of bank conflicts**, unless special memory layouts or warp-level instructions are used.

## Insights

1. Bank conflict behavior depends on memory layout, not on logical matrix shape
2. Shared memory is a low-latency but manually managed resource
3. Many "magic" GEMM optimizations arise naturally from avoiding bank conflicts
4. Tensor Core kernels rely on warp-level register tiling to bypass these issues
