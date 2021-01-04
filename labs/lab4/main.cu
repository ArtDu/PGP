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


__global__ void kernel_change_curr_line(double *A, int i, int j, int m) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;                    // Абсолютный номер потока
    int offset = blockDim.x * gridDim.x;                                // Общее кол-во потоков

    for (int p = i + 1 + idx; p < m; p += offset)
        A[j * m + p] /= A[j * m + i];
}

__global__ void kernel_change_other_lines(double *A, int i, int j, int n, int m, int EPS) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;                    // Абсолютный номер потока
    int idy = blockDim.y * blockIdx.y + threadIdx.y;                    // Абсолютный номер потока
    int offsetx = blockDim.x * gridDim.x;                                // Общее кол-во потоков
    int offsety = blockDim.y * gridDim.y;                                // Общее кол-во потоков

    for (int k = idy; k < n; k += offsety) {
        if (k != j && abs(A[k * m + i]) > EPS) {
            for (int p = i + 1 + idx; p < m; p += offsetx)
                A[k * m + p] -= A[j * m + p] * A[k * m + i];
        }
    }
}



const double EPS = 1E-9;

int compute_rank(host_vector<double> &A, int n, int m) {
    device_vector <double> d_A;

    int rank = 0;
    vector<bool> row_selected(n, false);
    for (int i = 0; i < m; ++i) {
        int j;
        for (j = 0; j < n; ++j) {
            if (!row_selected[j] && abs(A[j * m + i]) > EPS)
                break;
        }

        if (j != n) {
            ++rank;
            row_selected[j] = true;
            d_A = A;
            double *ptr_A = thrust::raw_pointer_cast(d_A.data());
            kernel_change_curr_line<<<64, 64>>> (ptr_A, i, j, m);
            CSC(cudaGetLastError());
            kernel_change_other_lines<<<dim3(32,32), dim3(32,32)>>> (ptr_A, i, j, n, m, EPS)
            CSC(cudaGetLastError());

            A = d_A;
//            for (int p = i + 1; p < m; ++p)
//                A[j * m + p] /= A[j * m + i];
//                A[j][p] /= A[j][i];
//            for (int k = 0; k < n; ++k) {
//                if (k != j && abs(A[k * m + i]) > EPS) {
//                    for (int p = i + 1; p < m; ++p)
//                        A[k * m + p] -= A[j * m + p] * A[k * m + i];
////                        A[k][p] -= A[j][p] * A[k][i];
//                }
//            }
        }
    }
    return rank;
}


int main(int argc, char *argv[]) {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    int n, m;
    cin >> n >> m;
    host_vector<double> h_A(n * m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            cin >> h_A[i * m + j];
        }
    }
    cout << compute_rank(h_A, n, m) << "\n";

    return 0;
}

