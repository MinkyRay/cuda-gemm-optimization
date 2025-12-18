# High-Performance CUDA GEMM: A Micro-Architectural Approach

![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?style=flat-square&logo=nvidia)
![Platform](https://img.shields.io/badge/Platform-RTX%204050%20(Ada)-blue?style=flat-square)
![Performance](https://img.shields.io/badge/Peak%20Performance-~7010%20GFLOPS-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> **"Optimization is not about writing complex code; it is about understanding the hardware constraints and systematically removing bottlenecks."**

## 📖 Overview

This repository documents the step-by-step optimization journey of Single-Precision General Matrix Multiplication (SGEMM) on the NVIDIA Ada Lovelace architecture. 

Starting from a naive implementation, we systematically apply micro-architectural optimizations—**Shared Memory Tiling**, **Register Blocking (Outer Product)**, **Register Blocking (Inner Product)**, and **Double Buffering (Async Copy)**—to achieve **~78% of the theoretical peak performance**.

Crucially, this project goes beyond code. It includes detailed **Technical Reports** analyzing *why* these optimizations work, covering topics from Shared Memory Bank Conflicts to Instruction Level Parallelism (ILP).

---

## 🚀 Optimization Roadmap & Benchmarks

We break down the optimization process into 4 evolutionary stages. Each stage addresses a specific hardware bottleneck.

| Stage | Kernel | Bottleneck Solved | Principle | Performance |
| :--- | :--- | :--- | :--- | :--- |
| **01** | [`src/01_naive.cu`](./src/01_naive.cu) | Global Memory Bandwidth | **Baseline** | ~700 GFLOPS |
| **02** | [`src/02_tiled.cu`](./src/02_tiled.cu) | DRAM Latency | **Spatial Locality** (Shared Mem) | ~800 GFLOPS |
| **03** | [`src/03_register_outer.cu`](./src/03_register_outer.cu) | Shared Mem Latency / Low AI | **Temporal Locality** (Outer Product) | ~6400 GFLOPS |
| **04** | [`src/04_double_buffering.cu`](./src/04_double_buffering.cu) | Pipeline Stalls | **Latency Hiding** (Software Pipeline) | **~7010 GFLOPS** |

*(Note: Performance measured on RTX 4050 Laptop @ 4096 x 4096)*

---

## 📚 Technical Knowledge Base (Deep Dives)

This is the core value of this repository. The following documents analyze the underlying mechanics of GPU optimization.

### 🧠 Core Analysis (Must Read)
* [**Sensitivity Analysis: The Impact of Register Block Size**](./docs/sensitivity_analysis.md) 🔥  
    *A rigorous study on the "Inverted-U" performance curve. Analyzes the trade-off between **ILP** (Heavy Threads) and **Occupancy** (Light Threads), and demonstrates the catastrophic effect of Register Spilling.*
* [**Outer Product vs. Inner Product**](./docs/outer_product_analysis.md)  
    *Why does changing the calculation order boost performance by 8x? A deep dive into **Arithmetic Intensity** and why register reuse is the key to breaking the memory wall.*

### 🛠️ Memory Subsystem
* [**Shared Memory Bank Conflict Analysis**](./docs/bank_conflict.md)  
    *Understanding memory banking, strides, and how padding resolves serialization penalties.*
* [**Vectorized Loading & Index Mapping**](./docs/index_mapping.md)  
    *How to utilize `float4` instructions and handle complex thread-to-data mapping logic.*

### ⚙️ Pipeline & Architecture
* [**Double Buffering & Async Copy**](./docs/double_buffering.md)  
    *Hiding Global Memory latency by overlapping data transfer (LSU) with computation (ALU) using `cp.async`.*
* [**Decoupled Architecture**](./docs/decoupled_architecture.md)  
    *Separating memory producers from compute consumers to maximize pipeline efficiency.*

---

## 📊 Performance Visualizations

### 1. Optimization Trajectory
*(Place your step-by-step GFLOPS line chart here)*
`![Benchmark Graph](./assets/benchmark_graph.png)`

### 2. Sensitivity Analysis (The "Sweet Spot")
*(Place your 4x4 vs 8x8 vs 16x16 sensitivity graph here)*
`![Sensitivity Graph](./assets/sensitivity_graph.png)`

> **Key Insight:** Peak performance requires balancing **Instruction Level Parallelism (ILP)** and **Thread Level Parallelism (TLP)**. As shown in the sensitivity analysis, an $8 \times 8$ register tile achieves the optimal equilibrium, while $16 \times 16$ causes register spilling and performance collapse.

---

## 💻 Getting Started

### Prerequisites
* NVIDIA GPU (Compute Capability 8.0+ recommended)
* CUDA Toolkit 11.0+
* CMake 3.18+

### Build & Run
```bash
# Clone the repository
git clone https://github.com/MinkyRay/cuda-gemm-optimization.git]
cd cuda-gemm-optimization

# Build
mkdir build && cd build
cmake ..
make

# Run the sensitivity benchmark
./bin_sensitivity
