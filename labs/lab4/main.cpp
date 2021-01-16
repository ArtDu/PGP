#include <bits/stdc++.h>
#include <algorithm>
//#include <thrust/extrema.h>
//#include <thrust/device_vector.h>

using namespace std;


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
        if(j != n)
            cerr << "col: " << i << " row: " << j << " mx: " <<  abs(A[j][i]) << "\n\n";
        if (j != n) {
            ++rank;
            row_selected[j] = true;
            for (int p = i + 1; p < m; ++p)
                A[j][p] /= A[j][i];

            cerr << "change curr line\n";
            for (int k = 0; k < n; ++k) {
                for (int l = 0; l < m; ++l) {
                    cerr << A[k][l] << " ";
                }
                cerr << "\n";
            }
            cerr << "\n";


            for (int k = 0; k < n; ++k) {
                if (k != j && abs(A[k][i]) > EPS) {
                    for (int p = i + 1; p < m; ++p)
                        A[k][p] -= A[j][p] * A[k][i];
                }
            }
            cerr << "change other lines\n";
            for (int k = 0; k < n; ++k) {
                for (int l = 0; l < m; ++l) {
                    cerr << A[k][l] << " ";
                }
                cerr << "\n";
            }
            cerr << "\n";
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
            cerr << A[i][j] << " ";
        }
        cerr << "\n";
    }
    cout << compute_rank(A) << "\n";

    return 0;
}

