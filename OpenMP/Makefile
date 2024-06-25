GCC_VERSION = 14
CC = gcc-$(GCC_VERSION)
CFLAGS = -fopenmp -Wall

TARGETS = blelloch_scan mat_mul matrix_addition prefix_sum prefix_sum_hillis_steele sum_of_array transpose

all: $(TARGETS)

blelloch_scan: blelloch_scan.c
	$(CC) $(CFLAGS) -o blelloch_scan blelloch_scan.c

mat_mul: mat_mul.c
	$(CC) $(CFLAGS) -o mat_mul mat_mul.c

matrix_addition: matrix_addition.c
	$(CC) $(CFLAGS) -o matrix_addition matrix_addition.c

prefix_sum: prefix_sum.c
	$(CC) $(CFLAGS) -o prefix_sum prefix_sum.c

prefix_sum_hillis_steele: prefix_sum_hillis_steele.c
	$(CC) $(CFLAGS) -o prefix_sum_hillis_steele prefix_sum_hillis_steele.c

sum_of_array: sum_of_array.c
	$(CC) $(CFLAGS) -o sum_of_array sum_of_array.c

transpose: transpose.c
	$(CC) $(CFLAGS) -o transpose transpose.c

clean:
	rm -f $(TARGETS)