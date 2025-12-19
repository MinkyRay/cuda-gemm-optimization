#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 

#define TILE_SIZE 16
// PADDING method：TILE_SIZE + 1 to avoid "Shared Memory Bank Conflict"
#define PADDED_TILE_SIZE (TILE_SIZE + 1)



// CUDA Kernel: Tiled GEMM with Shared Memory Padding

__global__ void sgemm_tiled_padded(float *A, float *B, float *C, int M, int N, int K) {

    __shared__ float As[TILE_SIZE][PADDED_TILE_SIZE];// TILE_SIZE x PADDED_TILE_SIZE: pad 1 to avoid bank conflict in column access
    __shared__ float Bs[TILE_SIZE][PADDED_TILE_SIZE];

    // thread index
    int tx = threadIdx.x; // thread column within block
    int ty = threadIdx.y; // thread row within block

    // global index in C
    int row = blockIdx.y * TILE_SIZE + ty;  // row in C
    int col = blockIdx.x * TILE_SIZE + tx; // col in C

    // C[row, col] accumulate register
    float Cvalue = 0.0f;


    // MAIN LOOP：Block Accumulation Loop

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE; 
    
    //For Loop to accumulate blook result
    for (int t = 0; t < num_tiles; ++t){
        //Step1: loading data from Global memory to Shared memory
        //Step1.1: A to As
        int global_a_row = row;
        int global_a_col = t * TILE_SIZE + tx; // coalesced memory access along row
        if (global_a_row < M && global_a_col < K){
            As[ty][tx] = A[global_a_row*K + global_a_col]; // linearized 1D access
        }
        else{
            As[ty][tx] = 0.0f; // padding for boundary
        }

        //Step1.2: B to Bs
        int global_b_row = t * TILE_SIZE + ty;
        int global_b_col = col;// blockIdx.x * TILE_SIZE + tx : coalesced along row
        if (global_b_row < K && global_b_col < N){
            Bs[ty][tx] = B[global_b_row*N + global_b_col];
        }
        else{
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        //Step2: Calculation As*Bs
        for (int k_inner = 0; k_inner < TILE_SIZE; ++k_inner){
            Cvalue += As[ty][k_inner] * Bs[k_inner][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N){
        C[row*N + col] = Cvalue; // write final result
    }
}


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

    printf("Running Standard Tiled Padding GEMM Benchmark\n");
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


    dim3 block(16, 16); 
    dim3 grid((N + 16 - 1) / 16, (M + 16 - 1) / 16);
    
    printf("Kernel Config: Grid=(%d, %d), Block=(%d, %d)\n", grid.x, grid.y, block.x, block.y);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    sgemm_tiled_padded<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // Record
    cudaEventRecord(start);
    for(int i=0; i<5; i++) { 
        sgemm_tiled_padded<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
    
    if (correct) printf("RESULT: PASS\n");
    else printf("RESULT: FAIL (Expected %f, Got %f)\n", (float)K, h_C[0]);

    // Performance Calculation
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = (flops / (milliseconds / 1000.0)) / 1e9;
    printf("Time: %.3f ms\n", milliseconds);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("Comparison:\n");
    //printf("  Naive: ~700 GFLOPS\n");

    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
//This kernel demonstrates correctness of shared-memory tiling, but does not outperform naive kernel on RTX 4050 for problem size 1024³ due to synchronization overhead and strong cache behavior of naive kernel.
