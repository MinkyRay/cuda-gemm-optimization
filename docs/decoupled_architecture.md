# Optimized SGEMM on Ada Lovelace: A Study on Decoupled Access/Compute Architecture

> **Author:** Mingyu Lei  
> **Target Architecture:** NVIDIA Ada Lovelace (RTX 4050 Laptop)  
> **Performance:** ~7000 GFLOPS (Peak)

---

## 1. Abstract

This project implements a highly optimized **Single-Precision General Matrix Multiplication (SGEMM)** kernel using CUDA. By evolving from a naive implementation to a highly tuned Register-Blocked and Vectorized version, we achieved significant performance gains.

The core contribution of this implementation is the **Decoupled Access and Compute Strategy**, where threads play dual roles—as "Data Movers" maximizing DRAM bandwidth and as "Computers" maximizing Arithmetic Intensity. This document details the thread mapping logic that enables this efficiency.

---

## 2. Kernel Configuration

To balance **Occupancy** and **Instruction Level Parallelism (ILP)**, we selected the following tiling parameters based on extensive sensitivity analysis:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **BM** | 128 | Block Tile M dimension (Global $\to$ Shared) |
| **BN** | 128 | Block Tile N dimension (Global $\to$ Shared) |
| **BK** | 8 | K-dimension stride (Accumulation loop step) |
| **TM** | 8 | Register Tile M dimension (Shared $\to$ Register) |
| **TN** | 8 | Register Tile N dimension (Shared $\to$ Register) |
| **Threads** | 256 | Threads per Block |

---

## 3. The Core Mechanic: Decoupled Access & Compute

In high-performance CUDA kernels, a strict 1:1 mapping between "what a thread loads" and "what a thread computes" often leads to uncoalesced memory access.

**Our Solution:** We treat the 256 threads in a block as a flexible workforce. Their `threadIdx.x` (0 to 255) is mapped differently depending on the execution phase.

### Phase 1: The Loader (Maximizing Bandwidth)

* **Goal:** Load a $128 \times 8$ tile ($A_s$) and an $8 \times 128$ tile ($B_s$) from Global Memory to Shared Memory.
* **Constraint:** Global Memory favors linear, aligned, 128-bit access patterns (**Vectorized Coalescing**).

We use `float4` vectorization. Each thread loads one `float4` (4 floats).

* Total elements to load per tile: $128 \times 8 + 8 \times 128 = 2048$ floats.
* Total `float4` vectors: $2048 / 4 = 512$ vectors.
* **Capacity:** 256 threads $\times$ 2 loads/thread = 512 vectors. **Perfect match.**

#### The Mapping Logic (Code Analysis)

```cpp
int tid = threadIdx.x; // Linear ID: 0 ~ 255

// === Loading Matrix A (128 rows x 8 cols) ===
// Width is 8 floats = 2 float4s.
// Therefore, we need 2 threads to cover one row of A.
int load_a_row = tid / 2;       // Stride: every 2 threads handle 1 row
int load_a_col = (tid % 2) * 4; // Thread 0->0, Thread 1->4 (Vectorized Offset)

// === Loading Matrix B (8 rows x 128 cols) ===
// Width is 128 floats = 32 float4s.
// Therefore, we need 32 threads (1 Warp) to cover one row of B.
int load_b_row = tid / 32;       // Stride: every 32 threads handle 1 row
int load_b_col = (tid % 32) * 4; // 0, 4, 8 ... 124 (Vectorized Offset)
```

#### Visualizing the "Loader" Mapping

**1. Loading Matrix A ($128 \times 8$):**
We pair threads. Threads (0,1) load Row 0. Threads (2,3) load Row 1.

```text
      [Col 0-3]    [Col 4-7]
Row 0: Thread 0  |  Thread 1
Row 1: Thread 2  |  Thread 3
...
```
*Efficiency: Threads issue requests to physically adjacent addresses.*

**2. Loading Matrix B ($8 \times 128$):**
The tile B is wide (128 floats). One row contains 32 `float4` elements. Since a Warp has 32 threads, one Warp reads exactly one row.

```text
       [Col 0-3] [Col 4-7] ... [Col 124-127]
Row 0:  Tid 0     Tid 1    ...  Tid 31    <-- Warp 0 (Perfect Coalescing)
Row 1:  Tid 32    Tid 33   ...  Tid 63    <-- Warp 1
...
```
*Efficiency: This creates the ideal memory access pattern. The memory controller sees a single, massive, contiguous request for the entire line, saturating the memory bus bandwidth.*

### Phase 2: The Computer (Maximizing Reuse)

* **Goal:** Compute a $128 \times 128$ sub-matrix of C.
* **Constraint:** Maximize Register Reuse and Instruction Level Parallelism (ILP) via Outer Product.

Once data is in Shared Memory (which supports random access), we switch the thread mapping to a **2D logical grid**.

#### The Mapping Logic (Code Analysis)

```cpp
// === Computing C (128 x 128) ===
// We organize 256 threads into a 16x16 logical grid.
int thread_row = tid / 16; 
int thread_col = tid % 16;
```

#### Visualizing the "Computer" Mapping

* **Logical Grid:** $16 \times 16$ threads.
* **Work per Thread:** Each thread is responsible for an $8 \times 8$ sub-tile.
    1.  It reads 8 values from $A_s$ (into registers).
    2.  It reads 8 values from $B_s$ (into registers).
    3.  It performs an Outer Product ($8 \times 1 \times 1 \times 8$) to update 64 positions in the C matrix.

```text
Total C Block (128x128)
+-----------------------+
| T(0,0) | T(0,1) | ... |  <-- Each box is an 8x8 matrix of floats
|--------+--------+-----|      computed by ONE thread completely in registers.
| T(1,0) | T(1,1) | ... |
|   ...  |   ...  | ... |
+-----------------------+
```

---

## 4. Performance Analysis

Why is this efficient?

* **Bandwidth Saturation (The Loader):**
    By using `float4` and assigning 32 threads (1 Warp) to read contiguous rows of Matrix B, we achieve **Perfect Memory Coalescing**. We treat Global Memory as a stream of bytes rather than a 2D matrix structure during the loading phase.

* **Latency Hiding (The Computer):**
    The outer product calculation performs 64 FMA operations (Floating Point Multiply-Accumulate) for every 16 floats loaded from Shared Memory.
    * **Arithmetic Intensity:** 4.0 FLOPs/Load.
    This high intensity allows the ALU pipeline to stay busy while waiting for Shared Memory or Global Memory operations, effectively hiding latency.

* **Shared Memory Banking:**
    The $128 \times 8$ layout for $A_s$ and $8 \times 128$ for $B_s$ (transposed implicitly via the loading pattern) minimizes Shared Memory Bank Conflicts during the compute phase.

---

## 5. Sensitivity Results

* **Register Block Size:** Peaked at $8 \times 8$. Smaller blocks (e.g., $4 \times 4$) underutilized ILP. Larger blocks (e.g., $12 \times 12$) caused Register Spilling.
* **Block Size:** Peaked at $64 \times 64$ and $128 \times 128$. Larger blocks ($256 \times 256$) failed due to Shared Memory capacity limits per SM on the RTX 4050.

---

## 6. Conclusion

By strictly decoupling the logic for memory access (Linear/Vectorized) from the logic for computation (2D Tiled/Outer Product), this kernel achieves performance close to the hardware limit. The `tid` re-mapping strategy is the key enabler of this "Best of Both Worlds" approach.
