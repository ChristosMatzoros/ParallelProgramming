#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM 2000		// You can increase this value to watch the power of the GPU over the CPU!!!
#define TILE_WIDTH 32 

__global__ void mat_mul(const float *M, const float *N, float *P, int WIDTH) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float pval = 0.0f;

    for (int step = 0; step <= (WIDTH / TILE_WIDTH) ; step++) {
        // Load current block of input strip band to shared memory
        if (row < WIDTH && (step * TILE_WIDTH + tx) < WIDTH) {
            Mds[ty][tx] = M[row * WIDTH + (step * TILE_WIDTH + tx)];
        } else {
            Mds[ty][tx] = 0.0f;
        }

        if ((step * TILE_WIDTH + ty) < WIDTH && col < WIDTH) {
            Nds[ty][tx] = N[(step * TILE_WIDTH + ty) * WIDTH + col];
        } else {
            Nds[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            pval += Mds[ty][k] * Nds[k][tx];
        }

        __syncthreads();
    }

    if (row < WIDTH && col < WIDTH) {
        P[row * WIDTH + col] = pval;
    }
}

void sequential_mat_mul(const float *M, const float *N, float *P, int WIDTH) {
    for (int i = 0; i < WIDTH; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < WIDTH; ++k) {
                sum += M[i * WIDTH + k] * N[k * WIDTH + j];
            }
            P[i * WIDTH + j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    float *h_M, *h_N, *h_P, *h_P_seq, *d_M, *d_N, *d_P;
    int size = sizeof(float) * DIM * DIM;

    h_M = (float *)malloc(size);
    h_N = (float *)malloc(size);
    h_P = (float *)malloc(size);
    h_P_seq = (float *)malloc(size);

    for (int i = 0; i < DIM * DIM; ++i) {
        h_M[i] = 1.0f;
        h_N[i] = 1.0f;
    }

    printf("Memory allocation to GPU.\n");
    cudaMalloc((void **)&d_M, size);
    cudaMalloc((void **)&d_N, size);
    cudaMalloc((void **)&d_P, size);
    printf("Memory allocation to GPU completed.\n");

    printf("Memory transfer from Host to Device.\n");
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    printf("Memory transfer from Host to Device completed.\n");

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((DIM + TILE_WIDTH - 1) / TILE_WIDTH, (DIM + TILE_WIDTH - 1) / TILE_WIDTH);

    printf("Kernel invocation.\n");
    mat_mul<<<gridDim, blockDim>>>(d_M, d_N, d_P, DIM);

    cudaDeviceSynchronize();

    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);
    printf("Memory transfer from Device to Host completed.\n");

    printf("Performing sequential matrix multiplication for comparison...\n");
    sequential_mat_mul(h_M, h_N, h_P_seq, DIM);

    printf("Comparing results...\n");
    int match = 1;
    for (int i = 0; i < DIM * DIM; ++i) {
        if (abs(h_P[i] - h_P_seq[i]) > 1e-5) {
            printf("Results don't match at index %d: GPU = %f, CPU = %f\n", i, h_P[i], h_P_seq[i]);
            match = 0;
            break;
        }
    }

    if (match) {
        printf("Results match!!!\n");
    } else {
        printf("Results don't match!!!\n");
    }

    free(h_M);
    free(h_N);
    free(h_P);
    free(h_P_seq);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}

