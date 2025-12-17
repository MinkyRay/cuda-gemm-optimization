#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#ifdef _WIN32
#pragma warning(disable: 4819)
#endif

#ifndef TM
#define TM 8
#endif

#ifndef TN
#define TN TM
#endif

#ifndef BM
#define BM 128
#endif

#ifndef BN
#define BN BM
#endif

#define BK 8




#define THREADS_PER_DIM_M (BM / TM) 
#define THREADS_PER_DIM_N (BN / TN)


#define THREADS_PER_BLOCK (THREADS_PER_DIM_M * THREADS_PER_DIM_N) 


#if (BM % TM != 0 || BN % TN != 0)
#error "BM and BN must be multiples of TM and TN"
#endif


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

// ==========================================
// Kernel: sgemm_4050_prefetch
// ==========================================
__launch_bounds__(256, 2)
__global__ void sgemm_4050_prefetch(float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int M, int N, int K) {
    
    int tid = threadIdx.x;
    
    //Register
    float thread_results[TM * TN] = {0.0f};
    float reg_a[TM];
    float reg_b[TN];

    //new prefetch register（differ from one buffer method)
    float4 prefetch_a;
    float4 prefetch_b;

    //Shared Memory
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 3. Global Pointers (float4 vectorization)
    float4 *A_ptr = (float4 *)A;
    float4 *B_ptr = (float4 *)B;

    int A_block_offset = blockIdx.y * BM * K;
    int B_block_offset = blockIdx.x * BN;

    //Move Indices(a differ from b)
    int load_a_row = tid / 2;
    int load_a_col = (tid % 2) * 4;
    int load_b_row = tid / 32;
    int load_b_col = (tid % 32) * 4;
    
    //Compute Indices(differ from move indices)
    int thread_row = tid / 16;
    int thread_col = tid % 16;

    // ========================================================
    // PROLOGUE: preload Tile[0] (k=0)
    // ========================================================
    {
        prefetch_a = A_ptr[(A_block_offset + load_a_row * K + (0 + load_a_col)) / 4];
        prefetch_b = B_ptr[(load_b_row + 0) * (N/4) + (B_block_offset + load_b_col) / 4];
        
        // register -> Shared Memory
        As[load_a_row * BK + load_a_col + 0] = prefetch_a.x;
        As[load_a_row * BK + load_a_col + 1] = prefetch_a.y;
        As[load_a_row * BK + load_a_col + 2] = prefetch_a.z;
        As[load_a_row * BK + load_a_col + 3] = prefetch_a.w;

        Bs[load_b_row * BN + load_b_col + 0] = prefetch_b.x;
        Bs[load_b_row * BN + load_b_col + 1] = prefetch_b.y;
        Bs[load_b_row * BN + load_b_col + 2] = prefetch_b.z;
        Bs[load_b_row * BN + load_b_col + 3] = prefetch_b.w;
        
        __syncthreads();
    }

    // ========================================================
    // MAIN LOOP : compute Tile[k] while preload Tile[k+1]
    // ========================================================
    for (int k_idx = 0; k_idx < K; k_idx += BK) {
        
        // --- STEP 1: prefrtch instruction (Global -> Register) ---
        if (k_idx + BK < K) {
            int next_k = k_idx + BK;
            prefetch_a = A_ptr[(A_block_offset + load_a_row * K + (next_k + load_a_col)) / 4];
            prefetch_b = B_ptr[(load_b_row + next_k) * (N/4) + (B_block_offset + load_b_col) / 4];
        }
        // --- STEP 2: Compute Tile (Shared -> Register -> FMA) ---
        // at the same time prefetch_a/b are loaded from DRAM to register
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Load A/B from Shared to Reg
            #pragma unroll
            for (int i = 0; i < TM; ++i) reg_a[i] = As[(thread_row * TM + i) * BK + k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) reg_b[j] = Bs[k * BN + (thread_col * TN + j)];

            // FMA Compute
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    thread_results[i * TN + j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads(); 
        // --- STEP 3: Pipeline Update 
        // register → Shared Memory
        if (k_idx + BK < K) {
            As[load_a_row * BK + load_a_col + 0] = prefetch_a.x;
            As[load_a_row * BK + load_a_col + 1] = prefetch_a.y;
            As[load_a_row * BK + load_a_col + 2] = prefetch_a.z;
            As[load_a_row * BK + load_a_col + 3] = prefetch_a.w;

            Bs[load_b_row * BN + load_b_col + 0] = prefetch_b.x;
            Bs[load_b_row * BN + load_b_col + 1] = prefetch_b.y;
            Bs[load_b_row * BN + load_b_col + 2] = prefetch_b.z;
            Bs[load_b_row * BN + load_b_col + 3] = prefetch_b.w;
        }
        
        __syncthreads();
    }

    // write back to Global Memory(same as one buffer method)
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

int main(int argc, char const *argv[]) {
    int M = 4096;
    int N = 4096;
    int K = 4096;
    if (argc > 1) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }


    printf("Running RTX 4050 Prefetch Optimization\n");
    printf("Matrix Size: M=%d, N=%d, K=%d\n", M, N, K);

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Host Memory
    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);

    // Initialization
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
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    printf("Kernel Config: Grid=(%d, %d), Block=%d\n", grid.x, grid.y, block.x);
    printf("Block Tile: %dx%d, Register Tile: %dx%d\n", BM, BN, TM, TN); 
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    sgemm_4050_prefetch<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // Record
    cudaEventRecord(start);
    for(int i=0; i<5; i++) { 
        sgemm_4050_prefetch<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= 5.0f; 

    // Copy back
    CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    // Verify
    printf("Verifying...\n");
    bool correct = true;
    if (fabs(h_C[0] - (float)K) > 1e-4) correct = false;
    if (fabs(h_C[M*N/2] - (float)K) > 1e-4) correct = false;
    if (fabs(h_C[M*N-1] - (float)K) > 1e-4) correct = false;

    if (correct) printf("RESULT: PASS\n");
    else printf("RESULT: FAIL (Expected %f, Got %f)\n", (float)K, h_C[0]);

    // Performance Calculation
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = (flops / (milliseconds / 1000.0)) / 1e9;
    printf("Time: %.3f ms\n", milliseconds);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("Comparison Target: Previous version was ~6167 GFLOPS.\n");

    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
