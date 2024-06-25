#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

// Number of iterations for the workload
#define NUM_ITERATIONS 100000000

// Function for the thread to perform some work
void* thread_work(void* arg) {
    long long sum = 0;
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        sum += i;
    }
    printf("Thread sum: %lld\n", sum);
    return NULL;
}

int main() {
    pthread_t thread1, thread2;
    struct timespec start, end;
    double parallel_time, serial_time;
    
    // Start the timer for parallel execution
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Create two threads
    pthread_create(&thread1, NULL, thread_work, NULL);
    pthread_create(&thread2, NULL, thread_work, NULL);

    // Wait for the threads to finish
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    // End the timer for parallel execution
    clock_gettime(CLOCK_MONOTONIC, &end);
    parallel_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Parallel execution time: %f seconds\n", parallel_time);

    // Start the timer for serial execution
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Perform the same work serially
    thread_work(NULL);
    thread_work(NULL);

    // End the timer for serial execution
    clock_gettime(CLOCK_MONOTONIC, &end);
    serial_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Serial execution time: %f seconds\n", serial_time);

    return 0;
}
