# To run the executables: mpirun -n <number_of_processors> ./executable
CC = mpicc
CFLAGS = -Wall

TARGETS = all_to_all broadcast gather reduce scatter send_recv

all: $(TARGETS)

all_to_all: all_to_all.c
	$(CC) $(CFLAGS) -o all_to_all all_to_all.c

broadcast: broadcast.c
	$(CC) $(CFLAGS) -o broadcast broadcast.c

gather: gather.c
	$(CC) $(CFLAGS) -o gather gather.c

reduce: reduce.c
	$(CC) $(CFLAGS) -o reduce reduce.c

scatter: scatter.c
	$(CC) $(CFLAGS) -o scatter scatter.c

send_recv: send_recv.c
	$(CC) $(CFLAGS) -o send_recv send_recv.c

clean:
	rm -f $(TARGETS)
