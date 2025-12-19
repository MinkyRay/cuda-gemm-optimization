
# Micro-architectural Analysis: Outer Product vs. Inner Product in GEMM Optimization

## 1. Introduction

In the optimization of General Matrix Multiply (GEMM) on GPUs, **Register Blocking** is the decisive technique used to mitigate the bandwidth bottleneck of Shared Memory. When implementing register-level computation, two distinct loop organizations exist: the **Outer Product** approach and the **Inner Product** approach.

While both methods are mathematically equivalent, their performance characteristics on SIMT architectures diverge significantly. This document analyzes the micro-architectural reasons why the Outer Product is the dominant strategy for high-performance GEMM kernels.

---

## 2. Implementation Comparison

We consider a single thread responsible for computing a sub-tile of matrix $C$ with dimensions $TM \times TN$. The $K$-dimension block size is $BK$.

### 2.1 Approach A: Outer Product (Register Tiled)

In this approach, the loop over the $K$ dimension is placed at the **outermost** level. We load a slice of vectors from Shared Memory into registers and perform a full Cartesian product update.

**Core Strategy:** `Load Slice` $\rightarrow$ `Compute Outer Product` $\rightarrow$ `Accumulate`

```cpp
// Outer Product Implementation
// Optimized for Data Reuse and Pipelining
#pragma unroll
for (int k = 0; k < BK; ++k) {
    // 1. Prefetch data from Shared Memory to Registers
    // Only here do we issue memory load instructions (LDS)
    #pragma unroll
    for (int i = 0; i < TM; ++i) reg_a[i] = As[(thread_row * TM + i) * BK + k];
    #pragma unroll
    for (int j = 0; j < TN; ++j) reg_b[j] = Bs[k * BN + (thread_col * TN + j)];

    // 2. outer product algorithm
    // The ALU operates at peak throughput without memory stalls
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            // reg_a[i] is reused TN times
            // reg_b[j] is reused TM times
            thread_results[i * TN + j] += reg_a[i] * reg_b[j];
        }
    }
}
__syncthreads();
```

### 2.2 Approach B: Inner Product (Dot Product)
This approach follows the naive mathematical definition of matrix multiplication. For every element in C, we iterate through the K dimension.

**Core Strategy:** `For each C_ij` $\rightarrow$ `Loop K` $\rightarrow$ `Load` $\rightarrow$ `Compute`

```cpp
// Inner Product Implementation
// Latency Bound (Not Recommended)
#pragma unroll
for (int i = 0; i < TM; ++i) {
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
        float dot_prod = 0.0f;
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Every calculation triggers Shared Memory access
            float val_a = As[(thread_row * TM + i) * BK + k];
            float val_b = Bs[k * BN + (thread_col * TN + j)];
            
            // Execution stalls waiting for memory load
            dot_prod += val_a * val_b;
        }
        thread_results[i * TN + j] += dot_prod;
    }
}

```



## 3. Micro-architectural Analysis
We analyze the performance gap using specific metrics, assuming a standard tile size of TM=8, TN=8.

### 3.1 Arithmetic Intensity & Data Reuse
Performance on modern GPUs is fundamentally bound by the ratio of compute operations to memory operations.

| Metric | Inner Product | Outer Product |
| --- | --- | --- |
| **Loads from SMEM** | 2 per FMA | 16 per 64 FMAs |
| **Arithmetic Intensity** | **0.5 FLOPs/Load** | **4.0 FLOPs/Load** |
| **Bottleneck** | LSU (Load/Store Unit) | Compute (ALU) |

> **Insight:** The Outer Product achieves an **8x reduction** in Shared Memory bandwidth pressure. The high arithmetic intensity keeps the ALU saturated.

### 3.2 Instruction Level Parallelism (ILP) and Latency Hiding
GPU performance relies heavily on hiding memory latency, not just by warp switching (occupancy), but by ILP within a single thread.

#### Pipeline Visualization```mermaid
sequenceDiagram
    participant SMEM as Shared Memory
    participant RF as Register File
    participant ALU as FP32 Units

    Note over SMEM, ALU: Inner Product (Stall Heavy)
    loop Every k step
        SMEM->>RF: Load val_a (Latency ~20 cycles)
        SMEM->>RF: Load val_b 
        Note over ALU: [STALL] Waiting for operands...
        RF->>ALU: FMA (Compute)
    end

    Note over SMEM, ALU: Outer Product (Pipeline Saturated)
    SMEM->>RF: Load Vector A
    SMEM->>RF: Load Vector B
    loop 64 Iterations
        RF->>ALU: FMA 1
        Note right of ALU: ALU is busy...
        RF->>ALU: FMA 2
        Note right of ALU: ...hiding latency...
        RF->>ALU: FMA 64
    end


* **Inner Product (Dependency Chain):**
The instruction stream involves a Read-After-Write (RAW) dependency: `Load -> Stall -> Compute`. The compiler cannot fill the stall slots because the computation strictly depends on the immediately preceding load.
* **Outer Product (Pipelining):**
Once the vectors `reg_a` and `reg_b` are loaded into the Register File, the compiler issues a sequence of 64 independent FMA instructions. This long arithmetic chain effectively hides the latency of memory loads for the *next* iteration of k (in a Double Buffering context).

### 3.3 Cost Analysis: Register Pressure
The only trade-off is register usage.

* **Inner Product:** Requires  2 temporary registers.
* **Outer Product:** Requires TM + TN = 16 additional registers to buffer the operands.

## 4. Conclusion
- Given that modern architectures (e.g., NVIDIA Ada Lovelace) provide ample register file resources (up to 255 registers per thread), investing **16 registers** to gain an **800% improvement in bandwidth efficiency** and superior latency hiding is the optimal architectural decision.

- The Outer Product is, therefore, not just an option, but the **dominant strategy** for Register Blocking.





