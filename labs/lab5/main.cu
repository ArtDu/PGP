#include <stdio.h>
#include <limits.h>
#include <unistd.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace std;
using namespace thrust;

#define CSC(call)                           \
do {                                \
  cudaError_t res = call;                     \
  if (res != cudaSuccess) {                   \
    fprintf(stderr, "ERROR in %s:%d. Message: %s\n",      \
        __FILE__, __LINE__, cudaGetErrorString(res));   \
    exit(0);                          \
  }                               \
} while(0)

#define BUCKET_SIZE 1024

__global__ void bitonic_sort_step(int *dev_values, int j, int k, int swap) {
    __shared__ int shared[BUCKET_SIZE];

    unsigned int i, ixj; /* Sorting partners: i and ixj */
    i = threadIdx.x;

    if(swap && i >= k / 2){
        int diff = i - k / 2;
        shared[i] = dev_values[k - 1 - diff];
    } else {
        shared[i] = dev_values[i];
    }

    __syncthreads();

    ixj = i ^ j;

    /* The threads with the lowest ids sort the array. */
    if ((ixj) > i) {
        if ((i & k) == 0) {
            /* Sort ascending */
            if (shared[i] > shared[ixj]) {
                /* exchange(i,ixj); */
                int temp = shared[i];
                shared[i] = shared[ixj];
                shared[ixj] = temp;
            }
        }
        if ((i & k) != 0) {
            /* Sort descending */
            if (shared[i] < shared[ixj]) {
                /* exchange(i,ixj); */
                int temp = shared[i];
                shared[i] = shared[ixj];
                shared[ixj] = temp;
            }
        }
    }
    __syncthreads();
    dev_values[i] = shared[i];
}

__global__ void swap_half(int *dev_values, int sz) {
    for (int i = sz / 2, j = sz - 1; i < j; i++, j-- ){
        thrust::swap(dev_values[i], dev_values[j]);
    }
}

__global__ void swap_half_shared(int *dev_values, int sz) {
    __shared__ int shared[BUCKET_SIZE];

    unsigned int id;
    id = threadIdx.x;


    if(id >= sz / 2){
        int diff = id - sz / 2;
        shared[id] = dev_values[sz - 1 - diff];
    } else {
        shared[id] = dev_values[id];
    }

    __syncthreads();
    dev_values[id] = shared[id];
}

__global__ void test(int *dev_values, int sz) {
    dev_values[0] = sz;
}

void bitonic_merge(int *dev_values, int sz) {

//    test<<<1, 1>>>(dev_values, sz);
//    swap_half<<<1, 1>>>(dev_values, sz);
//    swap_half_shared<<<1, BUCKET_SIZE>>>(dev_values, sz);

    int j, k;
    /* Major step */
    k = sz;
    /* Minor step */
    j = k >> 1;
    bitonic_sort_step<<<1, sz>>>(dev_values, j, k, 1);
    for (; j > 0; j = j >> 1) {
        bitonic_sort_step<<<1, sz>>>(dev_values, j, k, 0);
    }

}

void bitonic(int *dev_values, int sz, int odd) {
    if(odd == 0) {
        for (int i = BUCKET_SIZE / 2; i + BUCKET_SIZE < sz; i += BUCKET_SIZE) {
            bitonic_merge(dev_values + i, BUCKET_SIZE);
        }
    } else {
        for (int i = 0; i < sz; i += BUCKET_SIZE) {
            bitonic_merge(dev_values + i, BUCKET_SIZE);
        }
    }
}

__global__ void oddeven_sort_step(int *values, int odd, int n) {
    __shared__ int shared[BUCKET_SIZE];

    int id = threadIdx.x;

    shared[id] = values[id];
    __syncthreads();

    if (odd == 0 && id % 2 == 0 && id + 1 < n) {
        if (shared[id] > shared[id + 1]) {
            int tmp = shared[id];
            shared[id] = shared[id + 1];
            shared[id + 1] = tmp;
        }
    }
    if (odd == 1 && id % 2 == 1 && id + 1 < n) {
        if (shared[id] > shared[id + 1]) {
            int tmp = shared[id];
            shared[id] = shared[id + 1];
            shared[id + 1] = tmp;
        }
    }

    __syncthreads();
    values[id] = shared[id];
}

void oddeven_sort(int *dev_values, int sz) {
    for (int i = 0; i < sz; i++) {
        oddeven_sort_step<<<1, sz>>>(dev_values, i % 2, sz);
    }
}

void parallel_sort(int *values, int sz) {
    int *dev_values;
    size_t size = sz * sizeof(int);

    CSC(cudaMalloc(&dev_values, size));
    CSC(cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice));

    // sort all buckets by odd-even sort
    for (int i = 0; i < sz; i += BUCKET_SIZE) {
        oddeven_sort(dev_values + i, BUCKET_SIZE);
    }


    for (int i = 0; i < 2 * (sz / BUCKET_SIZE) ; ++i){
        bitonic(dev_values, sz, i % 2);
    }
//    bitonic(dev_values, sz, 0);

    CSC(cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost));
    CSC(cudaFree(dev_values));
}

int main(int argc, char *argv[]) {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);



    int sz;
//    scanf("%d", &sz);
    fread(&sz, sizeof(int), 1, stdin);
    fprintf(stderr, "%d\n", sz);

    int new_sz = ceil((double)sz / BUCKET_SIZE) * BUCKET_SIZE;
//    fprintf(stderr, "%d\n", new_sz);

    int *values = (int*) malloc( new_sz * sizeof(int));
    fread(values, sizeof(int), sz, stdin);
    for (int i = sz; i < new_sz; ++i){
        values[i] = INT_MAX;
    }


//    int i;
//    for (i = 0; i < sz; ++i)
//        scanf("%d", &values[i]);
//    for(; i < new_sz; ++i)
//        values[i] = INT_MAX;



    parallel_sort(values, new_sz);


    fwrite(values, sizeof(int), sz, stdout);
//    for (int i = 0; i < sz; i++) {
//        fwrite(&values[i], sizeof(int), 1, stdout);
//        fprintf(stderr, "%d ", values[i]);
//    }
//    fprintf(stderr, "\n");
    free(values);
    return 0;
}

