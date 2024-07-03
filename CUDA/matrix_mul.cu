#include<cuda.h>
#include<stdio.h>
#define DIM 10000

__global__ void mat_mul(const float * A, const float * B, float *C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row<N && col<N){
        float temp=0;
        for (int k=0 ; k<N; k++){
            temp += A[row*N+k] * B[k*N+col];
        }
        C[row * N + col] = temp;
    }
}


int main(int srgc, char * argv[]){
    float * h_A, * h_B, * h_C, * d_A, * d_B, * d_C;
    int size = sizeof(float)*DIM*DIM;
    
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    for (int i=0; i<DIM*DIM; i++){
        h_A[i]=1.0;
        h_B[i]=1.0;
    }

    printf("Memory allocation to GPU.\n");
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    printf("Memory allocation to GPU completed.\n");
    
    printf("Memory transfer from Host to Device.\n");
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    printf("Memory transfer from Host to Device completed.\n");

    dim3 blockDim(256,256);
    dim3 gridDim((DIM+blockDim.x-1)/blockDim.x,(DIM+blockDim.y-1)/blockDim.y);
    
    printf("Kernel invocation.\n");
    mat_mul<<<gridDim,blockDim>>>(d_A, d_B, d_C, DIM);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("Memory transfer from Device to Host completed.\n");
    
    printf("Compare results...\n");
    int f=1;
    for (int i=0; i<DIM*DIM; i++){
        if(h_C[i]-DIM<0.00001){   // We are comparing floats!!
            continue;
        }else{
            printf("Results don't match!!!\n");
            f=0;
        }
    }
    
    if (f) printf("Results match!!!\n");
    
    free(h_A);
    free(h_B);
    free(h_C);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}