#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ==========================================
// RTX 4050 Tuned Parameters
// ==========================================
#define BM 128
#define BN 128
#define BK 8    
#define TM 8
#define TN 8


#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}


// Kernel: gemm_tiled_4x4_register_outer

__launch_bounds__(256, 2)
__global__ void gemm_tiled_8x8_register(float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int M, int N, int K) {
    
    int tid = threadIdx.x;
    
    // 1. Registers
    float thread_results[TM * TN] = {0.0f};
    float reg_a[TM];
    float reg_b[TN];

    // 2. Shared Memory
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 3. Global Pointers (float4)
    float4 *A_ptr = (float4 *)A;
    float4 *B_ptr = (float4 *)B;
    //blockIdx.y block left_top coordinate
    int A_block_offset = blockIdx.y * BM * K;
    int B_block_offset = blockIdx.x * BN;

    // 4. Load Indices
    int load_a_row = tid / 2;
    int load_a_col = (tid % 2) * 4;
    int load_b_row = tid / 32;
    int load_b_col = (tid % 32) * 4;
    
    // Thread coordinates for computation
    int thread_row = tid / 16;
    int thread_col = tid % 16;

    // Main Loop
    for (int k_idx = 0; k_idx < K; k_idx += BK) {
        
        // --- Vectorized Load ---
        float4 tmp_a = A_ptr[(A_block_offset + load_a_row * K + (k_idx + load_a_col)) / 4];
        float4 tmp_b = B_ptr[(load_b_row + k_idx) * (N/4) + (B_block_offset + load_b_col) / 4];
        //another version of tmp_a & tmp_b (sm_a_row = load_a_row, b/column the same)
        // float4 tmp_a = A_ptr[((blockIdx.y * BM + sm_a_row)* K + sm_a_col + k_idx)/4];
        // float4 tmp_b = B_ptr[((sm_b_row + k_idx)*N + blockIdx.x * BN + sm_b_col)/4];

        
        // Unpack to SMEM
        As[load_a_row * BK + load_a_col + 0] = tmp_a.x;
        As[load_a_row * BK + load_a_col + 1] = tmp_a.y;
        As[load_a_row * BK + load_a_col + 2] = tmp_a.z;
        As[load_a_row * BK + load_a_col + 3] = tmp_a.w;

        Bs[load_b_row * BN + load_b_col + 0] = tmp_b.x;
        Bs[load_b_row * BN + load_b_col + 1] = tmp_b.y;
        Bs[load_b_row * BN + load_b_col + 2] = tmp_b.z;
        Bs[load_b_row * BN + load_b_col + 3] = tmp_b.w;

        __syncthreads();

        // --- Compute ---
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) reg_a[i] = As[(thread_row * TM + i) * BK + k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) reg_b[j] = Bs[k * BN + (thread_col * TN + j)];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    thread_results[i * TN + j] += reg_a[i] * reg_b[j];
                }
            }
        }  
        __syncthreads();
    }

    // Write Back
    int global_c_row = blockIdx.y * BM + thread_row * TM;
    int global_c_col = blockIdx.x * BN + thread_col * TN;

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            if ((global_c_row + i) < M && (global_c_col + j) < N) {
                C[(global_c_row + i) * N + (global_c_col + j)] = thread_results[i * TN + j];
            }
        }
    }
}




// ==========================================
// Main Function
// ==========================================
int main(int argc, char const *argv[]) {

    int M = 4096;
    int N = 4096;
    int K = 4096;
    if (argc > 1) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    printf("Running on RTX 4050 Optimization Test\n");
    printf("Matrix Size: M=%d, N=%d, K=%d\n", M, N, K);

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Host Memory
    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);

    // Initialization (A=1, B=1 => Result should be K)
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;

    // Device Memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void **)&d_A, size_A));
    CHECK(cudaMalloc((void **)&d_B, size_B));
    CHECK(cudaMalloc((void **)&d_C, size_C));

    CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // Config
    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    printf("Kernel Config: Grid=(%d, %d), Block=%d\n", grid.x, grid.y, block.x);

    // Timing Setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = 0;
    bool correct = true;

    // =========================================================
    // TEST 1: Original Kernel (gemm_tiled_4x4_register)
    // =========================================================
    printf("\n>>> Testing Original Kernel: gemm_tiled_4x4_register\n");

    // Warmup
    gemm_tiled_8x8_register<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // Record
    cudaEventRecord(start);
    for(int i=0; i<5; i++) {
        gemm_tiled_8x8_register<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= 5.0f;

    // Copy back & Verify
    CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    
    correct = true;
    if (fabs(h_C[0] - (float)K) > 1e-4) correct = false;
    if (fabs(h_C[M*N/2] - (float)K) > 1e-4) correct = false;
    if (fabs(h_C[M*N-1] - (float)K) > 1e-4) correct = false;

    if (correct) printf("VERIFICATION: PASS\n");
    else printf("VERIFICATION: FAIL (Expected %f, Got %f)\n", (float)K, h_C[0]);

    gflops = (flops / (milliseconds / 1000.0)) / 1e9;
    printf("Time: %.3f ms\n", milliseconds);
    printf("Performance: %.2f GFLOPS\n", gflops);

    printf("\nNote: RTX 4050 Theoretical Peak (FP32) is approx 9000 GFLOPS.\n");

    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
