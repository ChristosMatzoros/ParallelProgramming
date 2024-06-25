#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>  // for sleep function

// Condition variable and mutex
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// Shared flag to indicate if the signal has been sent
int flag = 0;

void* thread1_func(void* arg) {
    // Lock the mutex
    pthread_mutex_lock(&mutex);

    // Wait for the signal if the flag is not set
    while (flag == 0) {
        printf("Thread 1 is waiting for the signal...\n");
        pthread_cond_wait(&cond, &mutex);
    }

    // The flag has been set, proceed
    printf("Thread 1 received the signal and is proceeding...\n");

    // Unlock the mutex
    pthread_mutex_unlock(&mutex);

    return NULL;
}

void* thread2_func(void* arg) {
    // Sleep for 2 seconds to simulate work
    sleep(5);

    // Lock the mutex
    pthread_mutex_lock(&mutex);

    // Set the flag and signal the condition variable
    flag = 1;
    printf("Thread 2 is sending the signal...\n");
    pthread_cond_signal(&cond);

    // Unlock the mutex
    pthread_mutex_unlock(&mutex);

    return NULL;
}

int main() {
    pthread_t thread1, thread2;

    // Create the threads
    pthread_create(&thread1, NULL, thread1_func, NULL);
    pthread_create(&thread2, NULL, thread2_func, NULL);

    // Wait for both threads to complete
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    // Destroy the mutex and condition variable
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);

    return 0;
}