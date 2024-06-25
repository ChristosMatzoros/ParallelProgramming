#include <omp.h>
#include <stdio.h>



int main(){
    int N = 5;
    // Initialize matrices
    int A[5][5] =  {{11,12,13,14,15},
                    {21,22,23,24,25},
                    {31,32,33,34,35},
                    {41,42,43,44,45},
                    {51,52,53,54,55}};

    int C[5][5] = {{0,0,0,0,0},
                    {0,0,0,0,0},
                    {0,0,0,0,0},
                    {0,0,0,0,0},
                    {0,0,0,0,0}};

    // Transpose in parallel
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
                C[i][j] = A[j][i];
        }
    }

    // Print the resulting matrix C
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }

    return 0;
}

