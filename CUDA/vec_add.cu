
#include <stdio.h>
#include <cuda.h>

__global__ void vec_add(const float * A, const float * B, float *C, int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (id<N) C[id] = A[id] + B[id];
    
}

int main(int argc, char *argv[]){
    float * h_A, * h_B, * h_C;
    float * d_A, * d_B, * d_C;
    int N = 100000;
    
    h_A = (float *)malloc(sizeof(float)*N);
    h_B = (float *)malloc(sizeof(float)*N);
    h_C = (float *)malloc(sizeof(float)*N);

    cudaMalloc((void**)&d_A,sizeof(float)*N);
    cudaMalloc((void**)&d_B,sizeof(float)*N);
    cudaMalloc((void**)&d_C,sizeof(float)*N);

    
    for(int i=0; i<N ; i++){
        h_A[i] = 1.0; 
        h_B[i] = 1.0;
    }
    
    cudaMemcpy(d_A, h_A, sizeof(float)*N,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float)*N,cudaMemcpyHostToDevice);
    
    int thread_per_block = 256;
    int blocks_per_grid = (N + thread_per_block - 1) / thread_per_block;
        
    vec_add<<<blocks_per_grid,thread_per_block>>>(d_A, d_B, d_C, N);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, sizeof(float)*N,cudaMemcpyDeviceToHost);
    
    for(int i=0; i<N ; i++){
        printf("%f ", h_C[i]); 
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}