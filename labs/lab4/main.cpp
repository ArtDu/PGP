#include <bits/stdc++.h>
#include <algorithm>
//#include <thrust/extrema.h>
//#include <thrust/device_vector.h>

using namespace std;

//#define CSC(call)                           \
//do {                                \
//  cudaError_t res = call;                     \
//  if (res != cudaSuccess) {                   \
//    fprintf(stderr, "ERROR in %s:%d. Message: %s\n",      \
//        __FILE__, __LINE__, cudaGetErrorString(res));   \
//    exit(0);                          \
//  }                               \
//} while(0)


//__global__ void kernel(double *data) {
//    int idy = blockIdx.y * blockDim.y + threadIdx.y;
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    int offsety = blockDim.y * gridDim.y;
//    int offsetx = blockDim.x * gridDim.x;
//
//}

const double EPS = 1E-9;

int compute_rank(vector <vector<double>> A) {
    int n = A.size();
    int m = A[0].size();

    int rank = 0;
    vector<bool> row_selected(n, false);
    for (int i = 0; i < m; ++i) {
        int j;
        for (j = 0; j < n; ++j) {
            if (!row_selected[j] && abs(A[j][i]) > EPS)
                break;
        }

        if (j != n) {
            ++rank;
            row_selected[j] = true;
            for (int p = i + 1; p < m; ++p)
                A[j][p] /= A[j][i];
            for (int k = 0; k < n; ++k) {
                if (k != j && abs(A[k][i]) > EPS) {
                    for (int p = i + 1; p < m; ++p)
                        A[k][p] -= A[j][p] * A[k][i];
                }
            }
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
    vector <vector<double>> A(n, vector<double>(m));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            cin >> A[i][j];
        }
    }
    cout << compute_rank(A) << "\n";

    return 0;
}

