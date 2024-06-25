#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


int main (){
    int N = 100;
    int * array1 = (int *)malloc(sizeof(int) * N);
    int * array2 = (int *)malloc(sizeof(int) * N);
    int * res = (int *)malloc(sizeof(int) * N);

    // Simple initialization of the array
    for (int i=1; i<=N; i++) {
        array1[i-1] = i;
        array2[i-1] = i;
    }

    // Produce the sum of the arrays in parallel
    #pragma omp parallel for
    for (int i=0;i<N;i++){
        res[i]= array1[i] + array2[i];
    }

    // Print the results
    for (int i=0; i<N; i++) {
        printf("%d ",res[i]);
    }

    printf("\n");

    // Free allocated memory
    free(array1);
    free(array2);
    free(res);

    return 0;
}