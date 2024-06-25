#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc,&argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);

    const int elements_per_proc = 4;
    int data[world_size * elements_per_proc];

    if (rank == 0) {
        // Initialize the array with some data
        for (int i = 0; i < world_size * elements_per_proc; i++) {
            data[i] = i;
        }
    }

    int recv_data[elements_per_proc];
    // Scatter the data from root process to all processes
    // MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
    MPI_Scatter(data, elements_per_proc, MPI_INT, recv_data, elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process prints its received data
    printf("Process %d received data:", rank);
    for (int i = 0; i < elements_per_proc; i++) {
        printf(" %d", recv_data[i]);
    }
    printf("\n");

    MPI_Finalize();
    return 0;
}