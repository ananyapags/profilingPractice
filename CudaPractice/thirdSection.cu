#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define SIZE 1024*1024*1024   // Number of elements in the vector
#define CHUNK_SIZE 1024*1024*128 // elements per chunk in this case there are 8 chunks, adjust based on avail host arch 
#define BLOCK_SIZE 1024 //Number of threads per block



/* ============================================================
   LESSON 1: THREADS, WARPS, AND BLOCKS
   
   
   Key ideas:
   - A warp consists of 32 threads
   - Threads are grouped into warps inside a block
   - Example:
       128 threads per block → 128 / 32 = 4 warps per block
   ============================================================ */
__global__ void test01() {

    // Compute the warp ID for this thread
    // Integer division groups threads into sets of 32
    int warpIdVal = threadIdx.x / 32;

    printf("Block ID: %d | Thread ID: %d | Warp ID: %d\n",
           blockIdx.x, threadIdx.x, warpIdVal);
}


/* ============================================================
   LESSON 2: VECTOR ADDITION (KERNEL)
   ------------------------------------------------------------
   Each thread computes one element of the output vector.

   Global index calculation:
     i = threadIdx.x + blockIdx.x * blockDim.x

   This allows the kernel to scale across multiple blocks.
   ============================================================ */

   /* ============================================================
   LESSON 3: VECTOR ADDITION WITH EXTRA-LARGE VECTORS
   ------------------------------------------------------------
   Let's do 4m elements in each vector rn
   ============================================================ */

__global__ void vecAdd(int *A, int *B, int *C, int chunkSize) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (index<chunkSize){
    // NOTE: No bounds check here — assumes enough threads are launched
    C[i] = A[i] + B[i];

    }
}

void randomInts(int* x, int size){
   for (int i = 0;i<size;i++){
      x[i] = rand()%100:
   }
}
/* ============================================================
   MAIN FUNCTION
   ============================================================ */
int main() {

    /* --------------------------------------------------------
       LESSON 1: KERNEL LAUNCH NOTES (test01)
       --------------------------------------------------------
       Kernel launch syntax:
         kernel<<<numBlocks, threadsPerBlock>>>();

       - Each block is assigned to an SM
       - Thread count is usually a power of 2
       - Max threads per block (Volta): 1024

       Example occupancy implications:
         1024 threads/block → max 2 blocks per SM
         512  threads/block → max 4 blocks per SM
       (assuming registers and shared memory allow it)

       Example launch (commented out):
         test01<<<2, 64>>>();
         cudaDeviceSynchronize();
       -------------------------------------------------------- */


    /* --------------------------------------------------------
       LESSON 2: VECTOR ADDITION (HOST CODE)
       -------------------------------------------------------- */

    int *A, *B, *C;       // Host pointers
    int *d_A, *d_B, *d_C; // Device pointers

    long int size = SIZE * sizeof(int);  // Total bytes per vector

    printf("\nHello 00\n");
    /* Step 1: Allocate host memory */
    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);

    printf("\nHello 01\n");

    /* Step 2: Allocate device memory */
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    printf("\nHello 02\n");

    
    /* Step 3: Initialize host vectors */
    for (int i = 0; i < SIZE; i++) {
        A[i] = i;
        B[i] = SIZE - i;
    }
    printf("\nHello 03\n");

    /* Step 4: Copy data from host to device */
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    printf("\nHello 04\n");

    /* Step 5: Set up timing events */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    /* Step 6: Launch vector addition kernel
       --------------------------------------------------------
       - Only grid size and block size are configurable
       - Total threads launched should cover all elements
       - Profilers are preferred over guesswork for tuning
       -------------------------------------------------------- */

    vecAdd<<<1024*432,1024>>>(d_A, d_B, d_C, SIZE);

    cudaEventRecord(stop);

    /* Step 7: Copy result back to host
       NOTE: cudaMemcpy direction matters */
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    /* Step 8: Measure execution time */
    cudaEventSynchronize(stop);
    float milli = 0.0f;
    cudaEventElapsedTime(&milli, start, stop);
    printf("Execution time: %f ms\n", milli);

    printf("\nFinished\n");

    /*
    Optional: verify results
    for (int i = 0; i < SIZE; i++) {
        printf("%d + %d = %d\n", A[i], B[i], C[i]);
    }
    */

    /* Step 9: Free memory */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);

    return 0;
}
