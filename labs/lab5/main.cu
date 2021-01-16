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

#define BLOCKS 1
#define BUCKET_SIZE 1024

__global__ void bitonic_sort_step(int *values, int sz, int odd) {
    __shared__ int shared[BUCKET_SIZE];

    int *dev_values, k = BUCKET_SIZE;
    unsigned int i, j, ixj, temp; /* Sorting partners: i and ixj */
    int block_id = blockIdx.x;
    int block_offset = gridDim.x;
    i = threadIdx.x;

    if(odd)
        for (int step = (BUCKET_SIZE / 2) + block_id * BUCKET_SIZE; step + BUCKET_SIZE < sz; step = step + block_offset * BUCKET_SIZE) {
            dev_values = values + step;

            if (i >= k / 2) {
                int diff = i - k / 2;
                shared[i] = dev_values[k - 1 - diff];
            } else {
                shared[i] = dev_values[i];
            }

            __syncthreads();

            for (j = k >> 1; j > 0; j = j >> 1) {
                ixj = i ^ j;
                /* The threads with the lowest ids sort the array. */
                if ((ixj) > i) {
                    if ((i & k) == 0) {
                        /* Sort ascending */
                        if (shared[i] > shared[ixj]) {
                            /* exchange(i,ixj); */
                            temp = shared[i];
                            shared[i] = shared[ixj];
                            shared[ixj] = temp;
                        }
                    }
                    if ((i & k) != 0) {
                        /* Sort descending */
                        if (shared[i] < shared[ixj]) {
                            /* exchange(i,ixj); */
                            temp = shared[i];
                            shared[i] = shared[ixj];
                            shared[ixj] = temp;
                        }
                    }
                }
                __syncthreads();
            }
            dev_values[i] = shared[i];
        }
    else
        for (int step = block_id * BUCKET_SIZE; step < sz; step = step + block_offset * BUCKET_SIZE) {
            dev_values = values + step;

            if (i >= k / 2) {
                int diff = i - k / 2;
                shared[i] = dev_values[k - 1 - diff];
            } else {
                shared[i] = dev_values[i];
            }

            __syncthreads();

            for (j = k >> 1; j > 0; j = j >> 1) {
                ixj = i ^ j;
                /* The threads with the lowest ids sort the array. */
                if ((ixj) > i) {
                    if ((i & k) == 0) {
                        /* Sort ascending */
                        if (shared[i] > shared[ixj]) {
                            /* exchange(i,ixj); */
                            temp = shared[i];
                            shared[i] = shared[ixj];
                            shared[ixj] = temp;
                        }
                    }
                    if ((i & k) != 0) {
                        /* Sort descending */
                        if (shared[i] < shared[ixj]) {
                            /* exchange(i,ixj); */
                            temp = shared[i];
                            shared[i] = shared[ixj];
                            shared[ixj] = temp;
                        }
                    }
                }
                __syncthreads();
            }
            dev_values[i] = shared[i];
        }

}

void bitonic_sort(int *values, int sz) {
    for (int _i = 0; _i < 2 * (sz / BUCKET_SIZE); ++_i) {
        bitonic_sort_step<<<BLOCKS, BUCKET_SIZE>>>(values, sz, _i % 2);
    }
}


__global__ void oddeven_sort(int *dev_values, int sz) {
    __shared__ int shared[BUCKET_SIZE];

    int id = threadIdx.x;
    int block_id = blockIdx.x;
    int block_offset = gridDim.x;
    int odd, i, n = BUCKET_SIZE;
    int *values;

    for (int j = block_id * BUCKET_SIZE; j < sz; j = j + block_offset * BUCKET_SIZE) {
        values = dev_values + j;
        shared[id] = values[id];
        __syncthreads();

        for (i = 0; i < n; i++) {
            odd = i % 2;
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
        }
        values[id] = shared[id];
    }


}


void parallel_sort(int *values, int new_sz) {
    int *dev_values;
    size_t size = new_sz * sizeof(int);

    CSC(cudaMalloc(&dev_values, size));
    CSC(cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice));

    // sort all buckets by odd-even sort
    oddeven_sort<<<BLOCKS, BUCKET_SIZE>>>(dev_values, new_sz);

    // sort buckets between themselves by odd-even merge sort
//    bitonic_sort<<<BLOCKS, BUCKET_SIZE>>>(dev_values, new_sz);
    bitonic_sort(dev_values, new_sz);


    CSC(cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost));
//    CSC(cudaFree(dev_values));
}

int main(int argc, char *argv[]) {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);



    int sz;
//    scanf("%d", &sz);
    fread(&sz, sizeof(int), 1, stdin);
//    fprintf(stderr, "%d\n", sz);

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

    cudaEvent_t start, stop;
    float gpu_time = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    parallel_sort(values, new_sz);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    cerr << "time:\n";
    cerr << "blocks = " << BLOCKS << "; threads = " << BUCKET_SIZE << "\n";
    cerr << gpu_time << endl;

    fwrite(values, sizeof(int), sz, stdout);
//    for (int i = 0; i < sz; i++) {
//        fwrite(&values[i], sizeof(int), 1, stdout);
//        fprintf(stderr, "%d ", values[i]);
//    }
//    fprintf(stderr, "\n");
//    free(values);
    return 0;
}

