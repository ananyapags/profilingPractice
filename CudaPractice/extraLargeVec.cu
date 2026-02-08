// How do we accommodate insanely big vectors? Lesson plus notes :)
#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>   // malloc, free, rand
#include <time.h>     // time (optional)

#define TOTAL_SIZE (1024LL*1024LL*1024LL)     // Number of elements in the full vector (1B ints)
#define CHUNK_SIZE (1024*1024*128)            // elements per chunk
#define BLOCK_SIZE 1024                       // threads per block

__global__ void test01() {
    int warpIdVal = threadIdx.x / 32;
    printf("Block ID: %d | Thread ID: %d | Warp ID: %d\n", blockIdx.x, threadIdx.x, warpIdVal);
}

__global__ void vecAdd(int *A, int *B, int *C, int chunkSize) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < chunkSize) {
        C[i] = A[i] + B[i];
    }
}

void randomInts(int* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = rand() % 100;}
    }


int main() {
    // (optional) seed rand
    srand((unsigned)time(NULL));

    int *d_A, *d_B, *d_C;                 // Device pointers
    int *chunk_A, *chunk_B, *chunk_C;     // Host pointers (chunk buffers)

    size_t chunkBytes = (size_t)CHUNK_SIZE * sizeof(int);

    /* Allocate device memory (for one chunk) */
    cudaMalloc((void**)&d_A, chunkBytes);
    cudaMalloc((void**)&d_B, chunkBytes);
    cudaMalloc((void**)&d_C, chunkBytes);

    /* Allocate host memory (for one chunk) */
    chunk_A = (int*)malloc(chunkBytes);
    chunk_B = (int*)malloc(chunkBytes);
    chunk_C = (int*)malloc(chunkBytes);

    int numBlocks = (CHUNK_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE;


    for (long long offset = 0; offset < TOTAL_SIZE; offset += CHUNK_SIZE) {

        // current chunk size (smaller only for last chunk)
        int currentChunkSize = (TOTAL_SIZE - offset) < CHUNK_SIZE ? (int)(TOTAL_SIZE - offset) : CHUNK_SIZE;

        printf("\nOffset %lld | currentChunkSize %d\n", offset, currentChunkSize);

        // random data for this chunk, intaliazate each chucnk and send htese values to the gpu and use cuda memcpy
        randomInts(chunk_A, currentChunkSize);
        randomInts(chunk_B, currentChunkSize);

        // copy chunk to GPU
        cudaMemcpy(d_A, chunk_A, (size_t)currentChunkSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, chunk_B, (size_t)currentChunkSize * sizeof(int), cudaMemcpyHostToDevice);

        // launch kernel
        vecAdd<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_C, currentChunkSize);

        // copy result back to host (Device -> Host), you can send htis anywhere
        cudaMemcpy(chunk_C, d_C, (size_t)currentChunkSize * sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < 10; i++) {
            printf("C[%d] = %d   (A=%d + B=%d)\n",
                   i, chunk_C[i], chunk_A[i], chunk_B[i]);
        }
        // (optional) quick sanity check
        // printf("C[0]=%d (A[0]=%d, B[0]=%d)\n", chunk_C[0], chunk_A[0], chunk_B[0]);
    }

    // free AFTER loop
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(chunk_A);
    free(chunk_B);
    free(chunk_C);

    return 0;
}
