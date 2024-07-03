#include <cuda.h>
#include <stdio.h>

#define N 100000000
#define BLOCK_SIZE 256

// CUDA kernel for sum reduction
__global__ void sum_reduction(const float *A, float *res, int size) {
    extern __shared__ float smem[];
    int thid = threadIdx.x;
    int id = blockIdx.x * blockDim.x + thid;

    // Load data into shared memory
    smem[thid] = (id < size) ? A[id] : 0.0f;
    __syncthreads();

    // Perform reduction in shared memory
    for (int step = blockDim.x / 2; step > 0; step /= 2) {
        if (thid < step) {
            smem[thid] += smem[thid + step];
        }
        __syncthreads();
    }

    // Use atomicAdd to combine results from each block
    if (thid == 0) {
        atomicAdd(res, smem[0]);
    }
}

// Function to check for CUDA errors
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    float *h_A, *d_A, *d_res, h_res = 0.0f;
    int size = N * sizeof(float);

    // Allocate host memory
    h_A = (float*)malloc(size);
    if (h_A == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f; // Example initialization
    }

    // Allocate device memory
    checkCudaError(cudaMalloc((void**)&d_A, size), "Failed to allocate device memory for d_A");
    checkCudaError(cudaMalloc((void**)&d_res, sizeof(float)), "Failed to allocate device memory for d_res");

    // Copy host array to device
    checkCudaError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "Failed to copy data from host to device");

    // Initialize device result to 0
    checkCudaError(cudaMemset(d_res, 0, sizeof(float)), "Failed to set device memory for d_res");

    // Kernel launch configuration
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sum_reduction<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_A, d_res, N);

    // Check for kernel launch errors
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    
    // Ensure the kernel has finished executing
    checkCudaError(cudaDeviceSynchronize(), "Failed to synchronize");

    // Copy result back to host
    checkCudaError(cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy data from device to host");

    // Check result
    printf("Sum: %f\n", h_res);

    // Clean up
    free(h_A);
    cudaFree(d_A);
    cudaFree(d_res);

    return 0;
}
