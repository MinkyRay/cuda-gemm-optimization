# Sensitivity Analysis of SGEMM Optimization on NVIDIA Ada Lovelace Architecture

![Performance](https://img.shields.io/badge/Peak%20Performance-~7007%20GFLOPS-red.svg?style=flat-square)
![Hardware](https://img.shields.io/badge/GPU-RTX%204050%20(Ada)-blue.svg?style=flat-square)
![Metric](https://img.shields.io/badge/Metric-Sensitivity%20Analysis-green.svg?style=flat-square)

> **Author:** Mingyu Lei  
> **Affiliation:** Department of Computational Mathematics, Nankai University  
> **Research Interests:** Computational Mathematics; GPU Architecture; High Performance Computing
## 1. Abstract

This report presents a rigorous empirical analysis of Single-Precision General Matrix Multiplication (SGEMM) performance on the NVIDIA Ada Lovelace architecture (RTX 4050 Laptop). Moving beyond simple optimization, this study investigates the non-linear relationship between hardware resource allocation and computational throughput.

We systematically evaluate the impact of **Optimization Strategies**, **Micro-Architectural Granularity (Register Blocking)**, and **Macro-Architectural Granularity (Block Tiling)**. Our results demonstrate that peak performance is achieved not by maximizing a single parameter, but by identifying the precise equilibrium point between **Instruction Level Parallelism (ILP)** and **Streaming Multiprocessor (SM) Occupancy**.

---

## 2. Performance Evolution: Breaking the Memory Wall

To establish a baseline, we analyze the performance trajectory across four distinct optimization stages. This progression highlights the shift of the system bottleneck from Memory Bandwidth to Compute Latency.

| Optimization Stage | Throughput (GFLOPS) | Speedup | Dominant Bottleneck Analysis |
| :--- | :--- | :--- | :--- |
| **Naive Baseline** | ~700 | 1.0x | **Memory Bound (DRAM).** Arithmetic Intensity is low (~0.25 FLOPs/Byte). The Compute Units (ALUs) are starved waiting for Global Memory. |
| **Shared Mem Tiling** | ~800 | 1.1x | **Latency Bound.** Reduces Global Memory traffic, but execution is stalled by Shared Memory access latency and address calculations. |
| **Register Blocking** | ~6400 | 9.1x | **Compute Bound.** Moves the innermost loop to Registers. High ILP effectively hides Shared Memory latency. |
| **Async Prefetching** | **~7007** | **10.0x** | **Pipeline Latency.** Uses Software Pipelining to overlap Global Memory data transfer with the computation of the current tile. |

**Key Insight:** The 8x performance leap from *Shared Mem Tiling* to *Register Blocking* confirms that on Ada architectures, **hiding instruction latency** via Register Reuse is as critical as managing memory bandwidth.

---

## 3. Micro-Architecture Sensitivity: Register Block Size

This section investigates the impact of thread-level work granularity, defined by the Register Tile dimensions ($TM \times TN$), on performance.

* **Controlled Variable:** Block Tile Size fixed at $128 \times 128$.
* **Independent Variable:** Register Tile Size ($2\times2$ to $16\times16$).

### 3.1 The "Inverted-U" Performance Curve

Our benchmark reveals a distinct performance curve governed by the trade-off between **ILP** and **Register Pressure**.

#### Regime I: Under-Utilization ($2\times2$, $4\times4$)
* **Performance:** Suboptimal (~600 - 4500 GFLOPS).
* **Analysis:** **Insufficient ILP.**
    * With small register tiles, the ratio of math instructions (FMA) to memory instructions (Load) is low.
    * Each thread issues too few independent instructions, preventing the Warp Scheduler from effectively covering the pipeline latency.

#### Regime II: The Sweet Spot ($8\times8$)
* **Performance:** **Peak (~7000 GFLOPS).**
* **Analysis:** **Optimal Balance.**
    * Each thread performs 64 FMA operations per iteration.
    * Register usage is high enough to maximize ILP but remains low enough (~80-100 registers/thread) to allow sufficient **Occupancy** (Active Warps per SM). This enables the hardware to perform context switching during long-latency memory stalls.

#### Regime III: Resource Exhaustion ($10\times10$, $12\times12$)
* **Performance:**  Collapse (< 2500 GFLOPS).
* **Analysis:** **Register Spilling & Occupancy Collapse.**
    * **Spilling:**  A $12\times12$ tile pushes register usage beyond the hardware limit per thread (255 registers). The compiler forces excess variables into **Local Memory** (high-latency DRAM), creating enormous artificial memory traffic.
    * **Low Occupancy:**  Even before spilling, the massive register footprint per thread depletes the SM's physical Register File (64K entries). This limits the number of active warps (Occupancy) to a critical low.
    * **Consequence:**  With too few active warps, the scheduler cannot switch contexts to hide **Global Memory Latency** (DRAM access stalls). The SM pipeline is left idle waiting for data, causing the performance cliff.

---

## 4. Macro-Architecture Sensitivity: Block Tile Size

This section investigates the impact of SM resource allocation, defined by the Shared Memory Block dimensions ($BM \times BN$), on performance.

* **Controlled Variable:** Register Tile fixed at optimal $8 \times 8$.
* **Independent Variable:** Block Tile Size ($32\times32$ to $256\times256$).

### 4.1 Trade-off: Data Reuse vs. SM Concurrency

#### Small Blocks ($32\times32$)
* **Performance:** ~3900 GFLOPS.
* **Analysis:** **Low Data Reuse.**
    * While this configuration enables high Occupancy (many blocks per SM), the small tile size results in redundant Global Memory accesses, increasing pressure on the DRAM bandwidth.

#### Optimal Blocks ($64\times64$)
* **Performance:** **Peak (~7010 GFLOPS).**
* **Analysis:** **High Concurrency.**
    * This configuration strikes the best balance. The Shared Memory footprint is modest (~8KB-16KB), allowing the SM scheduler to launch multiple blocks concurrently.
    * **High Active Block count** makes the kernel more resilient to memory latency bubbles.

#### Large Blocks ($128\times128$)
* **Performance:** High (~6450 GFLOPS), but slightly degraded compared to $64\times64$.
* **Analysis:** **Limited Concurrency.**
    * This configuration maximizes Data Reuse from Global Memory.
    * However, the large Shared Memory requirement reduces the number of Active Blocks that can fit on a single SM. With fewer warps available for scheduling, the system is less able to hide latency.

#### Oversized Blocks ($256\times256$)
* **Result:** **Kernel Launch Failure.**
* **Analysis:** **Hard Constraint Violation.**
    * The Shared Memory or Register requirement per block exceeds the physical capacity of the Ada Lovelace Streaming Multiprocessor (SM).

---

## 5. Conclusion

This study characterizes the performance landscape of SGEMM on the RTX 4050. The analysis leads to three critical conclusions for High-Performance Computing on Ada Lovelace:

1.  **Register Spilling is a Hard Cliff:** Performance does not degrade gracefully when register limits are exceeded; it collapses. Keeping register usage below the spilling threshold (<255) is the primary constraint.
2.  **Occupancy Matters:** While maximizing ILP (via large Register Blocks) is essential, it must not come at the cost of starving the SM of active warps. The $8\times8$ register tile proves to be the optimal compromise.
3.  **Concurrency Favors Medium Blocks:** While larger blocks theoretically improve data reuse, a $64\times64$ block size outperforms $128\times128$ by enabling higher block concurrency on the SM.

---

### Experimental Environment
* **GPU:** NVIDIA GeForce RTX 4050 Laptop
* **Architecture:** Ada Lovelace (Compute Capability 8.9)
* **Max Threads per SM:** 1536
* **L1/Shared Memory:** Configurable (up to 100KB on Ada)
