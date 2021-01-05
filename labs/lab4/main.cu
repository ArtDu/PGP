#include <bits/stdc++.h>
#include <algorithm>
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

typedef struct _comparator {
    __device__ bool operator()(double a, double b) {
        return fabs(a) < fabs(b);
    }
} comparator;

__global__ void kernel_swap_lines(double *A, int j, int n, int m, int mx_idx) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;                    // Абсолютный номер потока
    int offset = blockDim.x * gridDim.x;                                // Общее кол-во потоков
    double tmp;
    for (int k = idx; k < m; k += offset) {
        tmp = A[k * n + j];
        A[k * n + j] = A[k * n + mx_idx];
        A[k * n + mx_idx] = tmp;
    }

}

__global__ void kernel_change_curr_line(double *A, int j, int n, int m) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;                    // Абсолютный номер потока
    int offset = blockDim.x * gridDim.x;                                // Общее кол-во потоков

    for (int p = j + 1 + idx; p < m; p += offset)
        A[p * n + j] /= A[j * n + j];

}

__global__ void kernel_change_other_lines(double *A, int j, int n, int m) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;                    // Абсолютный номер потока
    int idy = blockDim.y * blockIdx.y + threadIdx.y;                    // Абсолютный номер потока
    int offsetx = blockDim.x * gridDim.x;                                // Общее кол-во потоков
    int offsety = blockDim.y * gridDim.y;                                // Общее кол-во потоков
//    double EPS = 1e-7;

    for (int k = j + 1 + idy; k < n; k += offsety) {
//        if(fabs(A[j * n + k]) > EPS)
            for (int p = j + 1 + idx; p < m; p += offsetx) {
                A[p * n + k] -= A[p * n + j] * A[j * n + k];
                // A[k][p] -= A[j][p] * A[k][i];
            }
    }
}



const int BLOCKS = 256;
const int THREADS = 256;
const double EPS = 1e-7;

int _compute_rank(double *A, int n, int m) {
    comparator comp;

    // copy to device, compute all only in device
    double *d_A;
    CSC(cudaMalloc(&d_A, sizeof(double) * n * m));
    CSC(cudaMemcpy(d_A, A, sizeof(double) * n * m, cudaMemcpyHostToDevice));

    int rank = 0;
    int mx_idx;
    device_ptr<double> data_ptr, mx_ptr;
    for (int j = 0; j < m; ++j) {

        mx_idx = j;
        data_ptr = device_pointer_cast(d_A + j * n); // get column begin
        mx_ptr = thrust::max_element(data_ptr + j, data_ptr + n, comp); // get max in column
        mx_idx = mx_ptr - data_ptr; // get max idx

//        cerr << "mx in " << j << " row(column): " << *mx_ptr << "\n\n";

        if (fabs(*mx_ptr) > EPS) {
            ++rank;

            if (j != mx_idx) { // swap
                kernel_swap_lines<<<BLOCKS, THREADS>>>(d_A, j, n, m, mx_idx);
                CSC(cudaGetLastError());
            }

//            cerr << "swap in " << j << " row(column): \n";
//            for (int i = 0; i < n; ++i) {
//                for (int j = 0; j < m; ++j) {
//                    cin >> A[j * n + i];
//                    cerr << A[j * n + i] << " ";
//                }
//                cerr << "\n";
//            }
//            cerr << "\n";

            kernel_change_curr_line<<<BLOCKS, THREADS>>>(d_A, j, n, m);
            CSC(cudaGetLastError());

//            cerr << "change curr line in " << j << " row(column): \n";
//            for (int i = 0; i < n; ++i) {
//                for (int j = 0; j < m; ++j) {
//                    cin >> A[j * n + i];
//                    cerr << A[j * n + i] << " ";
//                }
//                cerr << "\n";
//            }
//            cerr << "\n";

            kernel_change_other_lines<<<dim3(32, 32), dim3(32, 32)>>>(d_A, j, n, m);
            CSC(cudaGetLastError());

//            cerr << "change other line in " << j << " row(column): \n";
//            for (int i = 0; i < n; ++i) {
//                for (int j = 0; j < m; ++j) {
//                    cin >> A[j * n + i];
//                    cerr << A[j * n + i] << " ";
//                }
//                cerr << "\n";
//            }
//            cerr << "\n";

        }
    }
//    CSC(cudaFree(d_A));
    return rank;
}


int main(int argc, char *argv[]) {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    int n, m;
    cin >> n >> m;
    double *h_A = (double *)malloc(sizeof(double) * n * m);

    if(m > n) {
        cerr << "swap n and m:\n";
        swap(n, m);
        cerr << n << " " << m << "\n";
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                cin >> h_A[i * m + j];
//                if(i == 0 && j < 1000)
//                    cerr << h_A[i * m + j] << " ";
            }
//            cerr << "\n";
        }
    } else {
        cerr << n << " " << m << "\n";
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                cin >> h_A[j * n + i];
//                if(i == 0 && j < 1000)
//                    cerr << h_A[j * n + i] << " ";
            }
//            cerr << "\n";
        }
    }

//    cudaEvent_t start, stop;
//    float gpu_time = 0.0;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    cudaEventRecord(start, 0);

    int res = _compute_rank(h_A, n, m);
    cout << res << "\n";
    cerr << res << "\n";
//    cudaEventRecord(stop, 0);
//    cudaEventSynchronize(stop);
//    cudaEventElapsedTime(&gpu_time, start, stop);
//
//    cerr << "time:\n";
//    cerr << "blocks = " << BLOCKS << "; threads = " << THREADS << "\n";
//    cerr << "2d = 16x64 16x64\n";
//    cerr << gpu_time << endl;

//    free(h_A);
    return 0;
}

