import os
import subprocess
import re
import matplotlib.pyplot as plt
import sys

KERNELS = [
    {
        "name": "Naive (Baseline)",
        "file": "src/01_gemm_naive.cu", 
        "exe": "bin_naive"
    },
    {
        "name": "Shared Mem Tiled",
        "file": "src/02_gemm_tiled.cu", 
        "exe": "bin_tiled"
    },
    {
        "name": "Reg Block (Inner - Bad)",
        "file": "src/03_gemm_register_inner.cu", 
        "exe": "bin_reg_inner"
    },
    {
        "name": "Reg Block (Outer - Good)",
        "file": "src/03_gemm_register_outer.cu", 
        "exe": "bin_reg_outer"
    },
    # ---------------------
    {
        "name": "Double Buffering (Async)",
        "file": "src/04_gemm_double_buffering.cu", 
        "exe": "bin_reg_async"
    }
]


MATRIX_SIZES = [1024, 2048, 4096, 8192]


NVCC_FLAGS = "-arch=sm_89 -O3" 


def compile_kernels():
    print("========================================")
    print("Step 1: Compiling CUDA Kernels...")
    print("========================================")
    
    for k in KERNELS:
        src = k['file']
        exe = k['exe']
        if sys.platform == "win32":
            exe += ".exe"
            
        if not os.path.exists(src):
            print(f"[Error] Source file {src} not found!")
            continue

        cmd = f"nvcc {src} -o {exe} {NVCC_FLAGS}"
        print(f"Compiling {src} -> {exe} ...")
        
        ret = subprocess.call(cmd, shell=True)
        if ret != 0:
            print(f"[Error] Compilation failed for {src}")
            exit(1)
    print("Compilation finished successfully.\n")


def run_benchmark():
    results = {k['name']: [] for k in KERNELS}

    print("========================================")
    print("Step 2: Running Benchmarks...")
    print("========================================")

    for size in MATRIX_SIZES:
        print(f"\n--- Benchmarking Matrix Size: {size} x {size} ---")
        
        for k in KERNELS:
            exe = k['exe']
            if sys.platform == "win32":
                exe += ".exe"

            cmd = f"{exe} {size} {size} {size}"
            
            try:
                output = subprocess.check_output(cmd, shell=True).decode("utf-8", errors="ignore")
                
                match = re.search(r"Performance:\s+([\d\.]+)\s+GFLOPS", output)
                
                if match:
                    gflops = float(match.group(1))
                    results[k['name']].append(gflops)
                    print(f"[{k['name']}] -> {gflops:.2f} GFLOPS")
                else:
                    print(f"[{k['name']}] -> Parse Error (Check C++ output format)")
                    results[k['name']].append(0)
            
            except subprocess.CalledProcessError as e:
                print(f"[{k['name']}] -> Runtime Error")
                results[k['name']].append(0)

    return results


def plot_results(results):
    print("\n========================================")
    print("Step 3: Plotting Results...")
    print("========================================")
    
    plt.figure(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'D']
    
    for idx, (name, data) in enumerate(results.items()):
        plt.plot(MATRIX_SIZES, data, marker=markers[idx % len(markers)], linewidth=2, label=name)

    plt.title("RTX 4050 GEMM Performance Analysis", fontsize=16)
    plt.xlabel("Matrix Size (N)", fontsize=14)
    plt.ylabel("Performance (GFLOPS)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    

    plt.axhline(y=9000, color='r', linestyle='--', alpha=0.5, label='Theoretical Peak (~9000)')
    
    output_img = "gemm_benchmark_4050.png"
    plt.savefig(output_img)
    print(f"Graph saved to {output_img}")
    # plt.show()


if __name__ == "__main__":

    compile_kernels()

    data = run_benchmark()

    plot_results(data)
    
    print("\nBenchmark Suite Completed!")
