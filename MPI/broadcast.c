#include<stdio.h>
#include<mpi.h>
#include<string.h>

#define MAX 100

int main(int argc, char * argv[]){
    MPI_Init(&argc, &argv);
    int id;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        fprintf(stderr, "World size must be at least two for %s\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if(id == 0){
        char* message = "Hello from process 0";
        // int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
        MPI_Bcast(message, strlen(message) + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
        printf("Process 0 sent message: %s\n", message);
    } else if (id != 0) {
        char message[MAX];
        //int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
        MPI_Bcast(message, MAX, MPI_CHAR, 0, MPI_COMM_WORLD);
        printf("Process %d received message: %s\n", id, message);
    }

    MPI_Finalize();
    return 0;
}