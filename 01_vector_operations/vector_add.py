import numpy as np
from numba import cuda
import time

@cuda.jit
def vector_add(a, b, out):
    #if n > total_threads, one thread need to process more elements
    gdim = cuda.gridDim.x
    bdim = cuda.blockDim.x
    total_threads = gdim * bdim
    idx = cuda.grid(1)
    '''
    if idx < a.size:
        out[idx] = a[idx] + b[idx]
    '''
    for i in range(idx, a.size, total_threads):
        out[idx] = a[idx] + b[idx]
    

def cpu_vector_add(a, b):
    return a + b

def main():
    print("===== Vector Add (GPU vs CPU) Benchmark =====")
    

    n = 33_333_333
    print(f"Array size: {n:,}")
    

    np.random.seed(42)
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    

    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_out = cuda.device_array_like(a)
    
    # 4. 配置CUDA执行参数
    threads_per_block = 256  # 每个block的线程数，常用值：128, 256, 512
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    
    print(f"CUDA config: {blocks_per_grid} blocks x {threads_per_block} threads")
    
    # 5. 预热（首次运行可能有编译开销）
    vector_add[blocks_per_grid, threads_per_block](d_a, d_b, d_out)
    
    #GPU Version
    start_gpu = time.perf_counter()
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_out = cuda.device_array_like(a)
    start_kernel = time.perf_counter()
    vector_add[blocks_per_grid, threads_per_block](d_a, d_b, d_out)
    kernel_time = time.perf_counter() - start_kernel
    gpu_result = d_out.copy_to_host()
    gpu_time = time.perf_counter() - start_gpu
    
    #CPU Version
    start_cpu = time.perf_counter()
    cpu_result = cpu_vector_add(a, b)
    cpu_time = time.perf_counter() - start_cpu
    

    is_correct = np.allclose(gpu_result, cpu_result, rtol=1e-5)
    

    print("\n--- Results ---")
    print(f"CPU time: {cpu_time * 1000:.2f} ms")
    print(f"GPU time (incl. data copy): {gpu_time * 1000:.2f} ms")
    print(f"GPU kernel only: {kernel_time * 1000:.2f} ms") 
    
    if is_correct:
        speedup = cpu_time / gpu_time
        print(f"✓ Results match! Speedup: {speedup:.2f}x")

    else:
        print("✗ Error: GPU and CPU results differ!")
        print(f"Max difference: {np.max(np.abs(gpu_result - cpu_result))}")
    
    print("\n--- Sample Values (first 5) ---")
    print(f"Input a: {a[:5]}")
    print(f"Input b: {b[:5]}")
    print(f"CPU out: {cpu_result[:5]}")
    print(f"GPU out: {gpu_result[:5]}")

if __name__ == "__main__":
    main()
