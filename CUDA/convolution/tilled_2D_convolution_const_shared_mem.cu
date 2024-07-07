#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 1080
#define HEIGHT 720

#define FILTER_RADIUS 2
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2 * (FILTER_RADIUS))

__constant__ float F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

__global__ void convolution2D(float* In, float* Out, int width, int height) {
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    __shared__ float S[IN_TILE_DIM][IN_TILE_DIM];
    if (col >= 0 && col < width && row >= 0 && row < height) {
        S[threadIdx.y][threadIdx.x] = In[row * width + col];
    } else {
        S[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM) {
        if (col < width && row < height) {
            float Res = 0.0f;

            for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
                for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                    Res += F[fRow][fCol] * S[tileRow + fRow][tileCol + fCol];
                }
            }

            if (col < width && row < height) {
                Out[row * width + col] = Res;
            }
        }
    }
}

void convolution2DSequential(float* In, float* Out, float* filter, int width, int height) {
    int filterSize = 2 * FILTER_RADIUS + 1;
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float Res = 0.0f;
            for (int fRow = 0; fRow < filterSize; fRow++) {
                for (int fCol = 0; fCol < filterSize; fCol++) {
                    int inRow = row + fRow - FILTER_RADIUS;
                    int inCol = col + fCol - FILTER_RADIUS;
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                        Res += filter[fRow * filterSize + fCol] * In[inRow * width + inCol];
                    }
                }
            }
            Out[row * width + col] = Res;
        }
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        printf("CUDA Error: %s (%s)\n", cudaGetErrorString(err), msg);
        cudaDeviceReset();
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    cudaError_t err;
    int width = WIDTH;
    int height = HEIGHT;

    float* h_In = (float*)malloc(width * height * sizeof(float));
    float* h_Out = (float*)malloc(width * height * sizeof(float));
    float* h_OutSeq = (float*)malloc(width * height * sizeof(float));
    if (h_In == NULL || h_Out == NULL || h_OutSeq == NULL) {
        printf("Error allocating host memory!\n");
        return 1;
    }

    for (int i = 0; i < width * height; i++) {
        h_In[i] = 1.0f;
    }

    float fh[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];
    for (int i = 0; i < 2 * FILTER_RADIUS + 1; i++) {
        for (int j = 0; j < 2 * FILTER_RADIUS + 1; j++) {
            fh[i][j] = 1.0f;
        }
    }
    cudaMemcpyToSymbol(F, fh, (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * sizeof(float));

    float* d_In;
    float* d_Out;
    err = cudaMalloc(&d_In, width * height * sizeof(float));
    checkCudaError(err, "allocating device memory for input");

    err = cudaMalloc(&d_Out, width * height * sizeof(float));
    checkCudaError(err, "allocating device memory for output");

    err = cudaMemcpy(d_In, h_In, width * height * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError(err, "copying input data to device");

    dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM);
    dim3 dimGrid((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    convolution2D<<<dimGrid, dimBlock>>>(d_In, d_Out, width, height);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    checkCudaError(err, "running kernel");

    err = cudaMemcpy(h_Out, d_Out, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError(err, "copying output data to host");

    cudaFree(d_In);
    cudaFree(d_Out);

    convolution2DSequential(h_In, h_OutSeq, (float*)fh, width, height);

    bool match = true;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (abs(h_Out[i * width + j] - h_OutSeq[i * width + j]) > 1e-5) {
                match = false;
                break;
            }
        }
        if (!match) break;
    }

    printf("Convolution results match: %s\n", match ? "Yes" : "No");
    
    /*
    printf("GPU Output:\n");
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.3f ", h_Out[i * width + j]);
        }
        printf("\n");
    }

    printf("CPU Output:\n");
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.3f ", h_OutSeq[i * width + j]);
        }
        printf("\n");
    }
    */
    free(h_In);
    free(h_Out);
    free(h_OutSeq);

    return 0;
}