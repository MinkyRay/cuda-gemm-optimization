# Sensitivity Analysis of SGEMM Optimization on NVIDIA Ada Lovelace Architecture

![Performance](https://img.shields.io/badge/Peak%20Performance-~7007%20GFLOPS-red.svg?style=flat-square)
![Hardware](https://img.shields.io/badge/GPU-RTX%204050%20(Ada)-blue.svg?style=flat-square)
![Metric](https://img.shields.io/badge/Metric-Micro--Arch%20Analysis-green.svg?style=flat-square)

> **Author:** Mingyu Lei  
> **Affiliation:** Department of Computational Mathematics, Nankai University  
> **Research Interests:** Computational Mathematics; GPU Architecture; High Performance Computing

## 1. Abstract

This report presents a rigorous empirical analysis of Single-Precision General Matrix Multiplication (SGEMM) performance on the NVIDIA Ada Lovelace architecture (RTX 4050). Moving beyond simple heuristics, this study investigates the non-linear relationship between hardware resource allocation, instruction-level parallelism (ILP), and memory hierarchy.

We systematically evaluate the impact of **Micro-Architectural Patterns (Inner vs. Outer Product)**, **Register Granularity**, and **Block Tiling**. Our results demonstrate that peak performance is achieved not merely by maximizing resource usage, but by identifying the precise equilibrium governed by three fundamental principles: **Temporal Locality**, **Spatial Locality**, and **Latency Hiding**.

---

## 2. Performance Evolution: A Principle-Based Analysis

We analyze the performance trajectory across five distinct optimization stages. This progression highlights the shift of the system bottleneck from Memory Bandwidth to Compute Latency.

| Optimization Stage | Throughput (GFLOPS) | Speedup | Bottleneck & Principle Analysis |
| :--- | :--- | :--- | :--- |
| **Naive Baseline** | ~700 | 1.0x | **Bottleneck: Global Memory Bandwidth.** <br> • **Principle Failure:** Low *Arithmetic Intensity* (~0.25 FLOPs/Byte). <br> • **Note:** Performance is non-zero primarily because the NVCC compiler implicitly utilizes the L1 Cache, providing passive *Temporal Locality*. However, the ALUs are strictly stalled by DRAM latency. |
| **Shared Mem Tiled** | ~800 | 1.1x | **Bottleneck: Shared Memory Latency.** <br> • **Optimization:** Improves *Spatial Locality* by loading tiles into Shared Memory (SRAM). <br> • **Limitation:** While Tiling increases **Global Arithmetic Intensity** (reducing DRAM traffic by a factor of Tile Size), the **Local Arithmetic Intensity** (ALU ops per Shared Memory load) remains unchanged. Threads still fetch operands from memory for every FMA, keeping the pipeline latency-bound. |
| **Reg Block (Inner)** | ~3800 | 5.4x | **Bottleneck: Dependency Stalls & Low ILP.** <br> • **Method:** Inner Product (Dot Product) accumulation. <br> • **Flaw:** Requires frequent reduction operations on the same accumulator. This creates Read-After-Write (RAW) dependencies that prevent effective *Latency Hiding*. |
| **Reg Block (Outer)** | ~6400 | 9.1x | **Bottleneck: Compute Throughput (ALU).** <br> • **Method:** Outer Product accumulation. <br> • **Principle Success:** Maximizes *Temporal Locality* (Register Reuse) and *Latency Hiding* (High ILP). Data stays in registers for the entire inner loop, decoupling ALU throughput from Shared Memory latency. |
| **Async Prefetching** | **~7007** | **10.0x** | **Bottleneck: Pipeline Latency.** <br> • **Method:** Software Pipelining (Double Buffering). <br> • **Principle Success:** Converts "Wait for Global Memory" time into "Useful Computation" time. The Global-to-Register latency is completely overlapped by the ALU workload. |

### 2.1 The Critical Divergence: Inner vs. Outer Product

One of the most profound findings is the massive performance gap (~2600 GFLOPS) between the two Register Blocking strategies. This gap highlights a fundamental distinction in computer architecture: the difference between reducing **Physical Distance (Latency)** and increasing **Logical Reuse (Arithmetic Intensity)**.

* **Inner Product (The Trap of Physical Proximity):**
    * **Logic:** Calculates one output element $C_{ij}$ at a time by iterating through $K$ (Dot Product).
    * **Micro-Arch Failure:** Even though data is stored in Registers (the closest physical storage to the ALU), the **Local Arithmetic Intensity remains unchanged (1:1)**.
        * Every FMA instruction still requires two operand reads from the Register File. The access pattern is essentially "Load-Compute-Discard".
        * **Diagnosis:** The performance gain over Shared Memory Tiling comes solely from the lower latency of Registers compared to Shared Memory (Physical Distance), **not** from a reduction in memory traffic (Logical Reuse). The ALUs are still throttled by the instruction issue rate of memory operands.

* **Outer Product (The Triumph of Logical Reuse):**
    * **Logic:** Loads a vector of $A$ and a vector of $B$ into registers, then performs a rank-1 update on a tile of $C$.
    * **Micro-Arch Success:** This approach fundamentally changes the data lifecycle.
        * A loaded value is **not** discarded after one use; it is held in the register and reused against multiple elements of the other vector.
        * **Diagnosis:** This achieves a **True Increase in Arithmetic Intensity**. By decoupling the number of math operations from the number of load/store operations, we shift the kernel from being *Latency Bound* (waiting for data to arrive) to being *Compute Bound* (waiting for math to finish).



---

## 3. Micro-Architecture Sensitivity: Register Block Size

This section investigates the impact of thread-level work granularity, defined by the Register Tile dimensions ($TM \times TN$), on performance.

* **Controlled Variable:** Block Tile Size fixed at $64 \times 64$.
* **Independent Variable:** Register Tile Size ($4\times4$, $8\times8$, $16\times16$).

### 3.1 The "Inverted-U" Performance Curve

Our benchmark reveals a distinct performance curve governed by the trade-off between **Instruction Level Parallelism (ILP)** and **Register Pressure**.

#### Regime I: Under-Utilization ($4\times4$)
* **Performance:** Suboptimal (~4530 GFLOPS).
* **Principle Violation:** **Insufficient Latency Hiding.**
    * With small register tiles ($TM \times TN = 16$), the ratio of math instructions (FMA) to memory instructions (Load) is too low.
    * Each thread issues too few independent instructions between memory barriers, preventing the Warp Scheduler from effectively covering the pipeline latency bubbles.

#### Regime II: The Sweet Spot ($8\times8$)
* **Performance:** **Peak (~7010 GFLOPS).**
* **Equilibrium:** **Optimal Balance.**
    * **High ILP:** Each thread performs 64 FMA operations per iteration, providing a deep, dependency-free instruction queue for the scheduler.
    * **Healthy Occupancy:** Register usage (~80-100 registers/thread) is balanced. It allows enough Warps to remain active on the SM to perform context switching during long-latency memory stalls, while maximizing per-thread throughput.

#### Regime III: Resource Exhaustion ($16\times16$)
* **Performance:** **Catastrophic Collapse (~229 GFLOPS).**
* **Hardware Limit:** **Severe Register Spilling.**
    * **Hard Constraint Violation:** The $C$ matrix accumulators alone require $16 \times 16 = 256$ registers, which strictly exceeds the Ada architecture's hardware limit per thread (255 registers).
    * **The "DRAM" Penalty:** The compiler is forced to spill variables into **Local Memory**. This effectively degrades the register access speed to DRAM speed (Global Memory latency), causing performance to plummet by nearly 30x compared to the peak.

---

> **💡 Key Insight: The Double-Edged Sword of Thread Granularity**
>
> Increasing the thread weight (Register Block Size) creates a fundamental tension between ILP and TLP, characterized by two benefits and two penalties:
>
> **The Dual Benefits (Why Heavy Threads are Fast):**
> 1.  **Surface-to-Volume Optimization (Higher AI):** Larger tiles ($8\times8$ vs $4\times4$) exponentially increase data reuse within the register file, strictly increasing the Arithmetic Intensity.
> 2.  **Micro-Latency Hiding (High ILP):** Heavier threads generate a denser stream of independent math instructions, allowing a single Warp to self-hide Shared Memory latencies without needing to switch contexts.
>
> **The Dual Penalties (Why Heavy Threads Collapse):**
> 1.  **Hard Constraint Violation (Spilling):** If the thread becomes too heavy ($16\times16$), it hits the physical register limit (255), forcing data into slow Local Memory.
> 2.  **Macro-Latency Exposure (Low TLP):** Heavier threads consume more SM resources, reducing the total number of active Warps (Occupancy). With fewer Warps available to switch to, the SM loses the ability to hide the massive Global Memory latency (~400 cycles).
---



## 4. Macro-Architecture Sensitivity: Block Tile Size

This section investigates the impact of SM resource allocation ($BM \times BN$) on performance. The results contradict the common heuristic that "larger tiles are always better," revealing a subtle micro-architectural trade-off on Ada Lovelace.

* **Controlled Variable:** Register Tile fixed at optimal $8 \times 8$.
* **Independent Variable:** Block Tile Size ($32\times32$, $64\times64$, $128\times128$).

### 4.1 Results Analysis

| Block Size | Threads/Block | Performance (GFLOPS) | Analysis Summary |
| :--- | :--- | :--- | :--- |
| **$32 \times 32$** | 16 | ~3867 | **The "Half-Warp" Penalty.** With only 16 threads per block, the hardware executes a full 32-thread Warp but masks off 50% of the lanes. Half the compute capacity is wasted. |
| **$64 \times 64$** | 64 | **~7009 (Peak)** | **The Sweet Spot (Granularity).** Small blocks mitigate "Register File Fragmentation," allowing higher SM Occupancy compared to larger blocks. |
| **$128 \times 128$** | 256 | ~6457 | **Resource Fragmentation.** While data reuse is maximized, the "heavy" register footprint of a 256-thread block causes quantization loss in the Register File, reducing active warps. |

### 4.2 Deep Dive: Why did 64x64 beat 128x128?

The counter-intuitive victory of the smaller $64 \times 64$ block is driven by **Register File Fragmentation**.

1.  **High Register Pressure:** Our optimal $8 \times 8$ Register Blocking kernel consumes significantly high registers (~80-100 per thread) to maintain the outer product accumulators.
2.  **Granularity Issue:**
    * **$128 \times 128$ (256 threads):** A single block consumes ~25,000 registers. An SM with 64K registers can only fit **2 blocks** (512 active threads). This results in low occupancy (~33%), leaving the SM scheduler with fewer warps to hide latency.
    * **$64 \times 64$ (64 threads):** A single block consumes ~6,400 registers. An SM can fit **~10 blocks** (640 active threads). This **25% increase in Occupancy** provides better latency hiding capabilities, outweighing the benefits of improved global memory reuse.

**Conclusion:** On register-heavy kernels (like Outer Product GEMM), smaller Block Tiles ($64\times64$) are often superior because they pack more efficiently into the SM's limited Register File, preventing resource stranding.

#### Oversized Blocks ($256\times256$)
* **Result:** **Kernel Launch Failure.**
* **Analysis:** **Hard Constraint Violation.**
    * The Shared Memory or Register requirement per block exceeds the physical capacity of the Ada Lovelace Streaming Multiprocessor (SM).

---

## 5. Conclusion: The Trinity of Optimization

This study characterizes the performance landscape of SGEMM on the RTX 4050. The analysis leads to three critical conclusions regarding the "Trinity of Optimization Principles":

1.  **Temporal Locality (Register Reuse is King):** It is insufficient to merely move data to Shared Memory. Data must be staged in Registers (via **Outer Product**) to decouple compute from memory latency. The shift from Inner to Outer product (~2600 GFLOPS gain) is the single largest optimization factor.
2.  **Latency Hiding (ILP & Pipelining):** Hardware is asynchronous. Performance is maximized when we explicitly structure code (**Double Buffering**) so that "expensive" operations (Global Load) happen in the shadow of "cheap" operations (ALU Math).
3.  **Resource Balance (Occupancy):** Optimization is a constrained problem. Pushing Register or Shared Memory usage to the limit (Regime III) is detrimental. The optimal point ($8\times8$ Reg, $64\times64$ Block) is where **Instruction Level Parallelism** and **Thread Level Parallelism** coexist.

---

### Experimental Environment
* **GPU:** NVIDIA GeForce RTX 4050
* **Architecture:** Ada Lovelace (Compute Capability 8.9)
* **Compiler:** NVCC 12.x (`-O3 -arch=sm_89`)
* **Max Threads per SM:** 1536
