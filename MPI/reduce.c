#include<stdio.h>
#include<mpi.h>
#include<string.h>

int main(int argc, char * argv[]){
    MPI_Init(&argc, &argv);
    int id;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int send_val = id;
    int recv_val;

    //int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
    MPI_Reduce(&send_val, &recv_val, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (id == 0) {
        printf("Process 0 --> sum of ids values from the rest of processes: %d\n", recv_val);
    }

    MPI_Finalize();
    return 0;
}