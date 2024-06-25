#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int elements_per_proc = 1;
    int send_data[size * elements_per_proc];
    int recv_data[size * elements_per_proc];

    // Initialize the send_data array
    printf("Process %d send: ", rank);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < elements_per_proc; j++) {
            send_data[i * elements_per_proc + j] = rank * size * elements_per_proc + i * elements_per_proc + j;
            printf("%d ",send_data[i * elements_per_proc + j]);
        }
    }
    printf("\n");

    // All-to-All communication
    MPI_Alltoall(send_data, elements_per_proc, MPI_INT, recv_data, elements_per_proc, MPI_INT, MPI_COMM_WORLD);

    // Each process prints its received data
    printf("Process %d received:", rank);
    for (int i = 0; i < size * elements_per_proc; i++) {
        printf(" %d", recv_data[i]);
    }
    printf("\n");

    MPI_Finalize();
    return 0;
}