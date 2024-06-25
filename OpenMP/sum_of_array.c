#include <omp.h>
#include <stdio.h>

int main (){
    int int_arr[5] = {10,15,20,25,30};
    int size = sizeof(int_arr) / sizeof(int_arr[0]);
    int sum = 0;

    #pragma omp parallel for reduction (+:sum)
    for (int i=0; i<size; i++){
        int id = omp_get_thread_num();
        printf("id:%d, val:%d\n",id, int_arr[i]);

        sum+=int_arr[i];
    }

    printf("SUM: %d\n",sum);

    return 0;
}