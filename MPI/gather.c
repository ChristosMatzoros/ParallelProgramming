#include <stdio.h>
#include <mpi.h>

int main(int argc, char * argv[]){
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int elements_per_proc = 4;
    int data[elements_per_proc];
    int recv_data[size * elements_per_proc];   

    for (int i=0; i<elements_per_proc ; i++) 
        data[i] = rank * elements_per_proc + i;

    MPI_Gather(data, elements_per_proc, MPI_INT, recv_data, elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i=0; i< size * elements_per_proc ; i++) 
            printf("%d ",recv_data[i]);
        printf("\n");
    }

    MPI_Finalize();

    return 0;
}