#include <stdio.h>
#include <omp.h>
#include <string.h>

void parallel_prefix_sum(int* A, int* PS, int N) {
    // Step 1: Copy input to output
    memcpy(PS, A, sizeof(int) * N);

    // Up-sweep (Reduction) Phase
    for (int step=1; step<N/2; step*=2){
        #pragma omp parallel for
        for (int i=N-1; i>0 ; i-= step*2){
            PS[i] += PS[i-step];
        }
    }

    // Clear the last element for exclusive scan
    PS[N - 1] = 0;

    // Down-sweep Phase
    for (int step=N/2; step>0; step/=2){
        #pragma omp parallel for
        for (int i=N-1; i>0 ; i-= step*2){
            int temp = PS[i-step];
            PS[i-step] = PS[i];
            PS[i] += temp;
        }
    }
}

int main() {
    int N = 8;
    int A[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int PS[8] = {0};

    parallel_prefix_sum(A, PS, N);

    // Print result
    for (int i = 0; i < N; i++) {
        printf("%d ", PS[i]);
    }
    printf("\n");

    return 0;
}