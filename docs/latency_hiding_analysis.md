# Demystifying Latency Hiding: A Micro-architectural Perspective on GPU Optimization

## 1. Introduction: The "Memory Wall" Challenge
In GPU high-performance computing, the fundamental bottleneck is rarely the peak FLOPs of the ALU, but rather the **Latency** of the memory hierarchy. Optimization is the art of "hiding" this latency—ensuring the ALUs are never idle while waiting for data.

This report summarizes my findings on the two pillars of latency hiding—**TLP** and **ILP**—validated through SGEMM benchmarks on the **NVIDIA Ada Lovelace (RTX 4050)** architecture.

---

## 2. TLP (Thread-Level Parallelism): The "Macro" Hide
TLP is the GPU’s primary defense against long-latency operations.

* **Primary Target:** **Global Memory (DRAM)** access (Typical: **200 - 600 cycles**).
* **Mechanism:** **Zero-cost Context Switching**. When a Warp is stalled by a Global Load (LSU), the Warp Scheduler immediately switches execution to another "ready" Warp in the same SM.
* **Physical Bottleneck:** **SM Resource Capacity**. It is limited by the **Register File** and **Shared Memory** size per SM, which determines the **Occupancy**.
* **Optimization Strategy:** Keep the thread footprint "light" to maximize the number of active Warps.

---

## 3. ILP (Instruction-Level Parallelism): The "Micro" Hide
ILP focuses on filling "bubbles" within a single thread's execution stream.

* **Primary Target:** **Shared Memory (SRAM)** access (Typical: **20 - 30 cycles**) and **Arithmetic Pipeline** latency (Typical: **4 - 8 cycles**).
* **Mechanism:** **Issue-Execution Decoupling**. The scheduler issues multiple independent instructions from a single Warp before needing the result of the first one.
* **Physical Bottleneck:** **Scoreboard & Data Dependency**. The hardware tracks register readiness; if the instruction stream lacks independent operations, the pipeline stalls due to Read-After-Write (RAW) dependencies.
* **Optimization Strategy:** **Loop Unrolling** and **Register Blocking (Outer Product)** to create long sequences of independent FMA instructions.



---

## 4. The Grand Trade-off: Occupancy vs. ILP
My experiments on the RTX 4050 revealed a non-linear relationship between thread granularity and performance:

| Metric | Regime I (Light) | Regime II (Sweet Spot) | Regime III (Heavy) |
| :--- | :--- | :--- | :--- |
| **Config (TMxTN)** | $4 \times 4$ | $8 \times 8$ | $16 \times 16$ |
| **Primary Hide** | High TLP | Balanced TLP/ILP | High ILP (attempted) |
| **Throughput** | ~4530 GFLOPS | **~7010 GFLOPS** | **~229 GFLOPS** |
| **Analysis** | High Occupancy, but low AI. | Optimal balance. | **Register Spilling.** Hard limit (255 regs) exceeded. |

**Key Finding:** Increasing thread weight (ILP) improves data reuse and masks Shared Memory latency, but eventually "crushes" TLP by exhausting registers. Peak performance occurs at the **minimum occupancy required to hide the latency**, rather than the maximum occupancy possible.

---

## 5. Advanced Technique: Double Buffering
Double Buffering is a strategic use of ILP to hide **Global Memory** latency—a task usually reserved for TLP.

* **Concept:** Utilizing **Software Pipelining**. While calculating tile $k$ using registers, we issue an asynchronous prefetch (`cp.async`) for tile $k+1$.
* **Hardware Impact:** It doubles the register pressure but effectively turns a **Memory-bound** kernel into a **Compute-bound** one by stretching the ILP chain across iterations.

---

## 6. Visualization of the Execution Pipeline

The GPU functions as an asynchronous dispatcher where the Warp Scheduler manages multiple independent execution pipes:



```text
[ Warp Scheduler ]
       |
       | (Checks Scoreboard: Is R0 ready?)
       |
   /---|---------------\
  |    |               |
  v    v               v
[ ALU Pipe ]     [ LSU Pipe ]     [ SFU Pipe ]
 (4-8 cycles)    (30-600 cycles)   (Special Func)
  |    |               |               |
  \----|---------------/               |
       | (Data Writeback)              |
[ Register File / Scoreboard Updates ] <---/
