# Optimizing GEMM with 4×4 Register Blocking: A CUDA Memory Hierarchy Study

![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green)
![Language](https://img.shields.io/badge/Language-C%2B%2B-blue)
![Platform](https://img.shields.io/badge/Platform-NVIDIA%20GPU-76B900)

## 1. Project Overview

This project implements a highly optimized General Matrix-Matrix Multiplication (GEMM) kernel using CUDA with **4×4 register blocking** and **shared memory tiling**.

The primary objective is to explore the intricate relationship between GPU memory hierarchy, thread-to-data mapping, and computational efficiency. By implementing register blocking at the thread level, we demonstrate how careful data placement across global memory, shared memory, and registers can dramatically improve performance while providing insights into GPU architecture fundamentals.

> **Note:** This implementation serves as a foundation for high-performance numerical linear algebra research and deep learning optimization, bridging the gap between theoretical parallel algorithms and practical hardware constraints.

---

## 2. Background

Matrix multiplication $C = A \times B$ is fundamental to scientific computing, machine learning, and computer graphics. On GPUs, optimizing GEMM involves navigating a complex memory hierarchy:

* **Global Memory**: High-latency, large capacity; requires coalesced access patterns.
* **Shared Memory**: Low-latency, limited capacity (per block); prone to bank conflicts.
* **Registers**: Lowest latency, thread-private; ideal for accumulating partial results.

### Computational Intensity

The ratio of arithmetic operations to memory operations determines performance. Traditional tiled GEMM implementations focus primarily on shared memory optimization. This project extends that approach by introducing **register blocking**, where each thread computes a small sub-matrix ($4 \times 4$ in this case) entirely within registers, further reducing shared memory traffic and improving computational intensity.

---

## 3. Kernel Design

### Thread Assignment Strategy

* Each thread computes a **4×4 sub-block** of the output matrix $C$.
* These 16 values are stored in **thread-private registers** throughout the computation.
* Threads are organized in blocks to match the register block size.

### Memory Hierarchy Utilization

1.  **Global → Shared**: $16 \times 16$ tiles of $A$ and $B$ are loaded cooperatively by threads in a block.
2.  **Shared → Registers**: Each thread loads elements from shared memory into registers for computation.
3.  **Register Accumulation**: Threads accumulate their $4 \times 4$ results entirely in registers.
4.  **Register → Global**: Final results are written back from registers to global memory.

### Thread-to-Data Mapping

```cpp
// Thread coordinates within block (0 ≤ tx, ty < 4)
int tx = threadIdx.x;
int ty = threadIdx.y;

// Position in shared memory (each thread handles 4×4 region)
int sm_col = 4 * tx;  // Starting column in shared memory
int sm_row = 4 * ty;  // Starting row in shared memory

// Global memory position of this thread's 4×4 block
int row = blockIdx.y * TILE_SIZE + sm_row;  // Starting row in global memory
int col = blockIdx.x * TILE_SIZE + sm_col;  // Starting column in global memory
```

### Memory Hierarchy Visualization

```text
┌─────────────────────────────────────────────────────────────┐
│                    Global Memory (A, B, C)                  │
│         High latency, large capacity, coalesced access      │
└───────────────────────────┬─────────────────────────────────┘
                            │ Load 16×16 tiles cooperatively
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                Shared Memory (As[TILE][TILE])               │
│          Low latency, block-shared, bank conflicts          │
└───────────────────────────┬─────────────────────────────────┘
                            │ Each thread loads needed elements
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Registers (c[4][4], a[4], b[4])            │
│        Lowest latency, thread-private, no conflicts         │
└───────────────────────────┬─────────────────────────────────┘
                            │ Accumulate 4×4 results
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Global Memory (C)                        │
│                 Write back final results                    │
└─────────────────────────────────────────────────────────────┘
```

### Thread/Block Data Mapping

```text
Block of 4×4 = 16 threads            Shared Memory 16×16 Tile
┌─────────────────┐                  ┌─────────────────────────┐
│ t00 t01 t02 t03 │ Thread Grid      │ ╔════╗ ╔════╗ ╔════╗ ...│
│ t10 t11 t12 t13 │ 4×4 arrangement  │ ║t00 ║ ║t01 ║ ║t02 ║    │
│ t20 t21 t22 t23 │                  │ ║4×4 ║ ║4×4 ║ ║4×4 ║    │
│ t30 t31 t32 t33 │                  │ ╚════╝ ╚════╝ ╚════╝    │
└─────────────────┘                  │ ╔════╗ ╔════╗ ╔════╗    │
                                     │ ║t10 ║ ║t11 ║ ║t12 ║    │
Thread t23's 4×4 register block:     │ ║4×4 ║ ║4×4 ║ ║4×4 ║    │
┌─────────────┐                      │ ╚════╝ ╚════╝ ╚════╝    │
│ c00 c01 c02 c03 │                  │ ...                     │
│ c10 c11 c12 c13 │                  └─────────────────────────┘
│ c20 c21 c22 c23 │
│ c30 c31 c32 c33 │
└─────────────┘
```

---

## 4. Implementation Details

### 1. Global Memory to Shared Memory

Threads cooperatively load $16 \times 16$ tiles from global memory into shared memory:

```cpp
// Loading A tile
int a_global_row = row + i;
int a_global_col = t * TILE_SIZE + sm_col + j;
As[sm_row + i][sm_col + j] = A[a_global_row * K + a_global_col];

// Loading B tile  
int b_global_row = t * TILE_SIZE + sm_row + i;
int b_global_col = col + j;
Bs[sm_row + i][sm_col + j] = B[b_global_row * N + b_global_col];
```

### 2. Shared Memory to Registers

Each thread loads elements needed for its $4 \times 4$ computation:

```cpp
float a_reg[4];  // Register storage for A elements
float b_reg[4];  // Register storage for B elements

// Load A elements (4 rows, same column k)
for (int i = 0; i < 4; ++i) {
    a_reg[i] = As[sm_row + i][k];
}

// Load B elements (4 columns, same row k)
for (int j = 0; j < 4; ++j) {
    b_reg[j] = Bs[k][sm_col + j];
}
```

### 3. Register Accumulation

Threads perform $4 \times 4$ matrix multiply-accumulate entirely in registers:

```cpp
float c_reg[4][4] = {{0.0f}};  // 4×4 accumulator in registers

// For each k in tile
for (int k = 0; k < TILE_SIZE; ++k) {
    // Load a_reg and b_reg (as above)
    // ...
    
    // 4×4 register-level GEMM
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            c_reg[i][j] += a_reg[i] * b_reg[j];
        }
    }
}
```

### 4. Registers to Global Memory

Final results are written from registers to global memory:

```cpp
for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
        if (row + i < M && col + j < N) {
            C[(row + i) * N + (col + j)] = c_reg[i][j];
        }
    }
}
```

---

## 5. Optimization Notes

### Memory Coalescing
* **Global Memory Reads**: Threads in a warp read consecutive addresses when loading tiles.
* **Problem**: For Matrix A, each thread reads 4 elements from the same row (potentially uncoalesced). For Matrix B, threads read different rows at the same column (broadcast pattern).
* **Solution**: Careful thread assignment and shared memory staging to ensure coalesced access.

### Shared Memory Bank Conflicts
* **Problem**: NVIDIA GPUs have 32 shared memory banks (4 bytes per bank). Multiple threads accessing the same bank causes serialized access.
* **Mitigation**: Padding shared memory arrays (`TILE_SIZE + 1`) to break bank alignment.

```cpp
// Reading As[sm_row + i][k] -> k varies, same row -> consecutive banks
// Reading Bs[k][sm_col + j] -> j varies, same column -> bank conflicts possible
```

### Register Pressure
Each thread uses approximately: `c[4][4]` (16 floats) + `a[4]` + `b[4]` = **24 float registers**.
* **Trade-off**: More registers lead to higher computational intensity but may cause register spilling to local memory if limits are exceeded.


### Computational Intensity Analysis
* **Operations per thread**: $4 \times 4 \times \text{TILE\_SIZE} \times 2 = 128 \times \text{TILE\_SIZE}$ FLOPs.
* **Memory operations**: $(4 \times 4 + 4 \times 4) \times \text{num\_tiles}$ loads from shared memory.
* **Result**: High Compute-to-Memory Ratio due to register blocking and reuse.
### Loop Unrolling

```cpp
#pragma unroll  // Compiler directive for performance
for (int i = 0; i < 4; ++i) {
    // ...
}
```

* **Benefits**: Reduces loop overhead, enables instruction-level parallelism.
* **Trade-off**: Increased code size, potential register pressure.

### Boundary Handling

```cpp
// Check bounds during loads
As[sm_row + i][sm_col + j] = (a0 + i < M && b0 + j < K) 
                            ? A[(a0 + i) * K + (b0 + j)] : 0.0f;
```

* **Approach**: Conditional loads with zero padding.

---

