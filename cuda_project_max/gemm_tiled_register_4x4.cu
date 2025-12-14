#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define TILE_SIZE 16           // Shared memory tile size
#define PADDED_TILE_SIZE (TILE_SIZE + 1) // Optional padding to avoid bank conflicts
#define REG_M 4                // Number of rows per thread register block
#define REG_N 4                // Number of columns per thread register block

// 4x4 register block tiled GEMM kernel
__global__ void sgemm_tiled_register_4x4(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    // Shared memory tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Step 0: Compute thread and block positions
    int tx = threadIdx.x;  // thread x within block
    int ty = threadIdx.y;  // thread y within block

    // Each thread handles a 4x4 sub-block in shared memory
    int sm_col = REG_N * tx;  // column offset in shared memory
    int sm_row = REG_M * ty;  // row offset in shared memory

    // Corresponding global memory positions (top-left of each 4x4 block)
    int row = blockIdx.y * TILE_SIZE + sm_row;
    int col = blockIdx.x * TILE_SIZE + sm_col;

    // Register accumulator for each thread's 4x4 sub-block
    float c[REG_M][REG_N] = {0.0f};

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Loop over tiles in K dimension
    for (int t = 0; t < num_tiles; ++t) {
        // Step 1.1: Load A tile from global memory to shared memory
        int global_a_row = row;
        int global_a_col = t * TILE_SIZE + sm_col;

        #pragma unroll
        for (int i = 0; i < REG_M; ++i) {
            #pragma unroll
            for (int j = 0; j < REG_N; ++j) {
                As[sm_row + i][sm_col + j] =
                    (global_a_row + i < M && global_a_col + j < K)
                    ? A[(global_a_row + i) * K + (global_a_col + j)]
                    : 0.0f;
            }
        }

        // Step 1.2: Load B tile from global memory to shared memory
        int global_b_row = t * TILE_SIZE + sm_row;
        int global_b_col = col;

        #pragma unroll
        for (int i = 0; i < REG_M; ++i) {
            #pragma unroll
            for (int j = 0; j < REG_N; ++j) {
                Bs[sm_row + i][sm_col + j] =
                    (global_b_row + i < K && global_b_col + j < N)
                    ? B[(global_b_row + i) * N + (global_a_col + j)]
                    : 0.0f;
            }
        }

        // Ensure all threads have loaded their tiles
        __syncthreads();

        // Step 2: Compute partial results in registers
        for (int k = 0; k < TILE_SIZE; ++k) {
            #pragma unroll
            for (int i = 0; i < REG_M; ++i) {
                #pragma unroll
                for (int j = 0; j < REG_N; ++j) {
                    c[i][j] += As[sm_row + i][k] * Bs[k][sm_col + j];
                }
            }
        }

        __syncthreads();
    }

    // Step 3: Write accumulated results from registers to global memory
    #pragma unroll
    for (int i = 0; i < REG_M; ++i) {
        #pragma unroll
        for (int j = 0; j < REG_N; ++j) {
            C[(row + i) * N + (col + j)] =
                ((row + i) < M && (col + j) < N) ? c[i][j] : 0.0f;
        }
    }
}



// Reference CPU implementation for verification
void cpu_gemm(const float* A, const float* B, float* C, 
              int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Validate GPU results
bool validate_results(const float* cpu_C, const float* gpu_C, 
                     int M, int N, float tolerance = 1e-3f) {
    int errors = 0;
    float max_error = 0.0f;
    
    for (int i = 0; i < M * N; ++i) {
        float diff = fabs(cpu_C[i] - gpu_C[i]);
        max_error = fmax(max_error, diff);
        if (diff > tolerance) {
            errors++;
            if (errors <= 10) {
                printf("  Mismatch at [%d]: CPU=%f, GPU=%f, diff=%f\n", 
                       i, cpu_C[i], gpu_C[i], diff);
            }
        }
    }
    
    if (errors > 0) {
        printf("  Found %d errors, max error = %f\n", errors, max_error);
        return false;
    }
    printf("  All results match! Max error = %f\n", max_error);
    return true;
}

int main() {
    // Matrix dimensions
    int M = 1024;
    int N = 1024;
    int K = 1024;
    
    printf("Matrix Multiplication Benchmark\n");
    printf("===============================\n");
    printf("Matrix size: %d x %d x %d\n", M, N, K);
    printf("Total operations: %.1f GFLOP\n", 2.0 * M * N * K / 1e9);
    
    // Allocate host memory
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C_cpu = (float*)malloc(sizeC);
    float* h_C_gpu = (float*)malloc(sizeC);
    
    // Initialize matrices
    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;  // All ones
    for (int i = 0; i < K * N; ++i) h_B[i] = 1.0f;  // All ones
    memset(h_C_cpu, 0, sizeC);
    memset(h_C_gpu, 0, sizeC);
    
    // Compute reference on CPU
    printf("\n[1] Computing reference on CPU...\n");
    cpu_gemm(h_A, h_B, h_C_cpu, M, N, K);
    printf("  CPU result C[0,0] = %f (expected %d)\n", h_C_cpu[0], K);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, sizeC);
    
    // Configure kernel launch
    dim3 block(4, 4);  // 4x4 = 16 threads? Wait...
    // Actually we need: tile_size=16, each thread computes 4x4
    // So each tile needs (16/4)*(16/4) = 4x4 = 16 threads
    // But your kernel uses sm_col = 4*tx, so tx max is TILE_SIZE/4 = 4
    // Similarly sm_row = 4*ty, so ty max is 4
    // So block(4, 4) = 16 threads is correct!
    
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, 
              (M + TILE_SIZE - 1) / TILE_SIZE);
    
    printf("\n[2] GPU Configuration:\n");
    printf("  Block size: (%d, %d) = %d threads\n", 
           block.x, block.y, block.x * block.y);
    printf("  Grid size: (%d, %d) = %d blocks\n", 
           grid.x, grid.y, grid.x * grid.y);
    printf("  Tiles per dimension: %d x %d\n", 
           (N + TILE_SIZE - 1) / TILE_SIZE, 
           (M + TILE_SIZE - 1) / TILE_SIZE);
    printf("  Shared memory per block: %.1f KB\n", 
           (2.0 * PADDED_TILE_SIZE * PADDED_TILE_SIZE * sizeof(float)) / 1024.0);
    
    // Warm up
    printf("\n[3] Warming up GPU...\n");
    sgemm_tiled_register_4x4<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error during warmup: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Benchmark
    printf("\n[4] Benchmarking GPU...\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int iterations = 100;
    float total_ms = 0.0f;
    
    for (int i = 0; i < iterations; ++i) {
        cudaEventRecord(start);
        sgemm_tiled_register_4x4<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
        
        // Check for errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error at iteration %d: %s\n", i, cudaGetErrorString(err));
            break;
        }
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    float avg_ms = total_ms / iterations;
    
    // Performance calculation
    float total_flops = 2.0f * M * N * K;
    float gflops = (total_flops / (avg_ms / 1000.0f)) / 1e9;
    float memory_bw = ((sizeA + sizeB + sizeC) / (avg_ms / 1000.0f)) / 1e9;
    
    printf("\n[5] Performance Results:\n");
    printf("  Average time: %.3f ms\n", avg_ms);
    printf("  Performance: %.2f GFLOP/s\n", gflops);
    printf("  Memory bandwidth: %.2f GB/s\n", memory_bw);
    printf("  Theoretical peak (RTX 4050): ~200-300 GFLOP/s\n");
    
    // Copy result back
    printf("\n[6] Validating results...\n");
    cudaMemcpy(h_C_gpu, d_C, sizeC, cudaMemcpyDeviceToHost);
    
    // Check a few values
    printf("  GPU result C[0,0] = %f (expected %d)\n", h_C_gpu[0], K);
    printf("  GPU result C[100,100] = %f (expected %d)\n", 
           h_C_gpu[100 * N + 100], K);
    printf("  GPU result C[%d,%d] = %f (expected %d)\n", 
           M-1, N-1, h_C_gpu[(M-1) * N + (N-1)], K);
    
    // Full validation
    bool valid = validate_results(h_C_cpu, h_C_gpu, M, N, 1e-3f);
    
    // Summary
    printf("\n[7] Summary:\n");
    if (valid) {
        printf("  ✓ Results are correct!\n");
        printf("  ✓ Performance: %.1f%% of theoretical peak\n", 
               (gflops / 250.0f) * 100.0f);  // Assuming 250 GFLOP/s peak
    } else {
        printf("  ✗ Results have errors!\n");
    }
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Reset device
    cudaDeviceReset();
}





