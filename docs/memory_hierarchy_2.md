# Deep Dive: Evolution of Memory Hierarchy and Data Proximity

In GPU optimization, performance is a function of **Data Proximity**. The central philosophy of our GEMM project is the systematic migration of compute-intensive data from high-latency, far-away storage to zero-latency, ALU-adjacent registers.

---

## 1. The Core Philosophy: "Migrating to the ALU"

The efficiency of a GEMM kernel is determined by how many arithmetic operations (FMA) we can perform per byte of data fetched from the "slow" memory. As we move down the hierarchy, the capacity shrinks, but the **Arithmetic Intensity (AI)** and **Reuse Factor** must increase exponentially.

| Tier | Resource | Latency (Cycles) | Capacity | Reuse Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **Far** | Global Memory | 400 - 600 | GBs | L2 Cache / Tiling |
| **Near** | Shared Memory | 20 - 30 | KBs | Block-wide Broadcast |
| **Closest** | Registers | ~0 (Effective) | Bytes/Thread | Register Blocking |

---

## 2. Tier 1: Global to Shared (The "Macro" Migration)

In a $1000 \times 1000$ GEMM, a naive implementation fetches data from Global Memory 1000 times for each element.

* **Evolution**: By introducing **Shared Memory Tiles** (e.g., $TILE\_SIZE = 64$), we load a tile once and reuse it across the thread block.
* **Impact**: Global fetches are reduced from 1000 to approximately 10-16 times (depending on Tile size).
* **Role of L2 Cache**: Redundant loads between different blocks are intercepted by the hardware-managed L2 Cache, further protecting the DRAM bus.



---

## 3. Tier 2: Shared to Register (The "Fine" Migration)

Registers are the only memory space directly connected to the ALU. Moving data from Shared Memory to Registers is the final step to break the "Memory Wall."

### 3.1 The "Inner Product" Approach (The Thread Granularity)
* **Logic**: Each thread handles a small sub-matrix but often re-reads elements from Shared Memory within the inner loop.
* **Constraint**: While it reduces Global traffic, it places heavy pressure on **Shared Memory Bandwidth**.

### 3.2 The "Outer Product" Approach (Register Blocking)
This is where the algorithm evolves. Instead of processing element-by-element, we perform a **Rank-1 Update**.

* **The Mapping Strategy**:
    1. Load a **Row Strip** of A ($TM$ elements) into registers.
    2. Load a **Column Strip** of B ($TN$ elements) into registers.
    3. Perform $TM \times TN$ FMA operations entirely within registers.
* **Data Perfection**: Once $A_{row}$ and $B_{col}$ are in registers, they are reused for every single accumulator in that step. This achieves **near-zero** Shared Memory traffic during the computation phase.



---

## 4. The Hard Limit: Register Pressure

If "closer is better," why not move the entire $64 \times 64$ tile into registers?

* **Resource Scarcity**: Each SM has a fixed Register File. On modern architectures like Ada Lovelace, the limit is **255 registers per thread**.
* **The "Performance Cliff"**: 
    * If we use too many registers (e.g., a $16 \times 16$ register tile), the compiler is forced to perform **Register Spilling**.
    * Data is "spilled" back to Local Memory (DRAM), causing performance to collapse (as seen in our benchmark: **~229 GFLOPS**).
* **Equilibrium**: The $8 \times 8$ configuration (64 accumulators) provides the optimal balance between high reuse and healthy SM **Occupancy**.

---

## 5. Summary: Proximity equals Performance

The evolution of our GEMM kernels demonstrates a clear trend:
1. **Naive**: Far from ALU, high latency, low throughput.
2. **Tiled**: Medium distance, reduced DRAM traffic.
3. **Register-Blocked (Outer Product)**: Closest to ALU, maximized data reuse, peak throughput.

**Final Insight**: The algorithm's innovation (Outer Product) is not just a mathematical change, but a strategic adaptation to the physical limitations of the GPU memory hierarchy.
