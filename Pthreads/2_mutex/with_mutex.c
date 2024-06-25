#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_ITERATIONS 50000

int counter = 0;
pthread_mutex_t mtx;

void* increment_counter(void* arg) {
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        pthread_mutex_lock(&mtx);
        counter++;
        pthread_mutex_unlock(&mtx);
    }
    return NULL;
}

int main() {
    pthread_t thread1, thread2;
    
    // int pthread_mutex_init(pthread_mutex_t *mutex, const pthread_mutexattr_t *attr);
    pthread_mutex_init(&mtx, NULL);

    // int pthread_create(pthread_t *thread, const pthread_attr_t *attr, void *(*start_routine)(void *), void *arg);
    pthread_create(&thread1, NULL, increment_counter, NULL);
    pthread_create(&thread2, NULL, increment_counter, NULL);

    // Wait for both threads to finish
    //int pthread_join(pthread_t thread, void **retval);
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    // Print the final value of the counter
    printf("Final counter value without mutex: %d\n", counter);

    pthread_mutex_destroy(&mtx);

    return 0;
}