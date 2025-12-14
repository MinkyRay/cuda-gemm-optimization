# GEMM: From CUDA Core to Tensor Core

## Motivation

This project studies the performance characteristics of CUDA GEMM implementations, with a particular focus on shared memory bank conflicts.

While many GEMM optimizations are often presented as implementation tricks, this repository aims to derive these optimizations from first principles, starting from naive CUDA kernels and analyzing how hardware-level constraints influence performance.

## Baseline: Naive CUDA GEMM

We begin with a naive CUDA GEMM implementation:

```cuda
__global__ void gemm_naive(float *A, float *B, float *C,
                           int M, int N, int K) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

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

This kernel is simple but severely bandwidth-limited.

## Shared Memory Tiling

We then introduce shared memory tiling:

```cuda
#define TILE_SIZE 16

__global__ void gemm_tiled(float *A, float *B, float *C,
                           int M, int N, int K) {

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float tmp = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        As[ty][tx] = (row < M && t * TILE_SIZE + tx < K)
                     ? A[row * K + t * TILE_SIZE + tx] : 0.0f;

        Bs[ty][tx] = (t * TILE_SIZE + ty < K && col < N)
                     ? B[(t * TILE_SIZE + ty) * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
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

On NVIDIA GPUs:

- Shared memory has 32 banks
- Each bank services 4 bytes
- Bank index is computed as:

```
bank_id = (address_in_bytes / 4) % 32
```

Thus, a float (4 bytes) maps exactly to one bank.

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

## Limitations & Next Step (Tensor Core)

This project does not yet use Tensor Cores.

The next step is to study:

- Warp-level matrix fragments
- `ldmatrix` instructions
- Tensor Core data layouts and their interaction with shared memory
- WMMA (Warp Matrix Multiply Accumulate) API
- Performance comparison between CUDA Core and Tensor Core implementations
