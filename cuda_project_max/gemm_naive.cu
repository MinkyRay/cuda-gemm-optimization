#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#ifdef _WIN32
#pragma warning(disable: 4819)
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

__global__ void sgemm_naive(float *A, float *B, float *C, int M, int N, int K) {
    //step0: define global index
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //step1: calculation
    if (row < M && col < N) {
        float tmp = 0.0f;
        for (int k = 0; k < K; ++k) {
            // ------------------------------------------------------------------
            // [Bottleneck Location: High Global Memory Traffic within the Loop]
            // Bottleneck: Excessive Global Memory (DRAM) accesses inside the inner loop.
            // 
            // - Arithmetic Intensity (AI) is extremely low: ~1/4 FLOPs/Byte. 
            //   The kernel is severely MEMORY BOUND.
            // - Data reuse is poor: For each C[row, col] calculation, 2*K single-precision floats 
            //   are fetched from the slow Global Memory.
            // - Although B access is coalesced, A access is broadcast.
            // ------------------------------------------------------------------
            tmp += A[row * K + k] * B[k * N + col];
        }

    
        C[row * N + col] = tmp; // write final result
    }
}


// Host Code
// ==========================================
// Main Function
// ==========================================
int main(int argc, char const *argv[]) {

    int M = 4096;
    int N = 4096;
    int K = 4096;
    if (argc > 1){
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    printf("Running Naive SGEMM Benchmark\n");
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


    dim3 block(32, 32); 
    dim3 grid((N + 32 - 1) / 32, (M + 32 - 1) / 32);
    
    printf("Kernel Config: Grid=(%d, %d), Block=(%d, %d)\n", grid.x, grid.y, block.x, block.y);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    sgemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // Record
    cudaEventRecord(start);

    for(int i=0; i<2; i++) { 
        sgemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= 2.0f; // Average

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

    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}

//Naive Kernel Performance Bottleneck AnalysisBottleneck Type: Memory BoundThe performance of this Naive GEMM Kernel is critically low, achieving less than 1% of the GPU's theoretical peak performance (based on the 3.48ms runtime). The primary bottleneck is the inefficient and excessive reliance on Global Memory (DRAM) access.Extremely Low Arithmetic Intensity (AI):Inside the critical inner loop, for every two floating-point operations (one multiply, one add), the kernel must fetch two single-precision floats (8 Bytes) from Global Memory.$$\text{AI} \approx \frac{2 \text{ FLOPs}}{8 \text{ Bytes}} = 0.25 \text{ FLOPs/Byte}$$This ratio is significantly lower than the computational throughput of modern GPUs, leaving the processing cores MEMORY BOUND and stalled (Data Starvation) waiting for data.Poor Data Locality and Reuse:Each thread performs $K$ loop iterations, repeatedly accessing the same row of matrix $A$ and the same column of matrix $B$. The lack of explicit local data storage means high latency Global Memory is accessed in every iteration, drastically limiting throughput.Inefficient Memory Access Pattern:While accessing matrix $B$ exhibits highly efficient Coalesced Access, the access pattern for matrix $A$ results in a Broadcast scenario where 32 threads read the exact same element. While not a conflict, this still represents an inefficient use of the massive available memory bandwidth for overall computation.Optimization Goal: The subsequent optimization must implement Shared Memory Tiling to transform the complexity of memory access from $O(N^3)$ to $O(N^2)$, dramatically increasing data locality and kernel throughput.
