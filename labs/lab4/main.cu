#include <stdio.h>
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

struct comparator
{
    __host__ __device__ bool operator()(double a, double b)
    {
        return fabs(a) < fabs(b);
    }
};


__global__ void kernel_swap_lines(double *A, int j, int n, int m, int mx_idx) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;                    // Абсолютный номер потока
    int offset = blockDim.x * gridDim.x;                                // Общее кол-во потоков
//    double tmp;
    for (int k = idx; k < m; k += offset) {
//        tmp = A[k * n + j];
//        A[k * n + j] = A[k * n + mx_idx];
//        A[k * n + mx_idx] = tmp;
        thrust::swap(A[k * n + j], A[k * n + mx_idx]);
    }
}


__global__ void kernel_change_lines(double *A, int row, int col, int n, int m) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;                    // Абсолютный номер потока
    int idy = blockDim.y * blockIdx.y + threadIdx.y;                    // Абсолютный номер потока
    int offsetx = blockDim.x * gridDim.x;                               // Общее кол-во потоков
    int offsety = blockDim.y * gridDim.y;                               // Общее кол-во потоков

    for (int k = row + idy + 1; k < n; k += offsety) {
            for (int p = col + idx + 1; p < m; p += offsetx) {
                A[p * n + k] -= ((A[p * n + row] / A[col * n + row]) * A[col * n + k]);
//                A[p * n + k] = A[col * n + k];
            }
    }
}



const int BLOCKS = 1024;
const int THREADS = 1024;
const double EPS = 1e-7;


int main(int argc, char *argv[]) {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    int n, m;
    scanf("%d%d", &n, &m);
    double *h_A = (double *)malloc(sizeof(double) * n * m);


//    cerr << n << " " << m << "\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            scanf("%lf", &h_A[j * n + i]);
//                cerr << h_A[j * n + i] << " ";
        }
//            cerr << "\n";
    }


//    cudaEvent_t start, stop;
//    float gpu_time = 0.0;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    cudaEventRecord(start, 0);



/////////////////////////////////////////////////////


    comparator comp;
    // copy to device, compute all only in device
    double *d_A;
    CSC(cudaMalloc(&d_A, sizeof(double) * n * m));
    CSC(cudaMemcpy(d_A, h_A, sizeof(double) * n * m, cudaMemcpyHostToDevice));

    int rank = 0;
    int mx_idx;
    double mx;
    device_ptr<double> data_ptr, mx_ptr;
    for (int j = 0, i = 0; j < m && i < n; ++j, ++i) {

        mx = EPS;
        data_ptr = device_pointer_cast(d_A + j * n); // get column begin
        mx_ptr = thrust::max_element(data_ptr + i, data_ptr + n, comp); // get max in column

        mx = fabs(*mx_ptr);
        mx_idx = mx_ptr - data_ptr; // get max idx

//        cerr << "(" << j << "," << i << "," << mx_idx << "); mx: " << mx << "\n\n";

        if (mx > EPS) {
            ++rank;

            if (i != mx_idx) { // swap
                kernel_swap_lines<<<BLOCKS, THREADS>>>(d_A, i, n, m, mx_idx);
                CSC(cudaGetLastError());
            }


            kernel_change_lines<<<dim3(16, 64), dim3(16, 64)>>>(d_A, i, j, n, m);
            CSC(cudaGetLastError());


        } else {
            --i;
        }
    }

//    CSC(cudaFree(d_A));
//    free(h_A);



    printf("%d", rank);
//    cerr << rank << "\n";

//    cudaEventRecord(stop, 0);
//    cudaEventSynchronize(stop);
//    cudaEventElapsedTime(&gpu_time, start, stop);
//
//    cerr << "time:\n";
//    cerr << "blocks = " << BLOCKS << "; threads = " << THREADS << "\n";
//    cerr << "2d = 32x32 32x32\n";
//    cerr << gpu_time << endl;




    return 0;
}

