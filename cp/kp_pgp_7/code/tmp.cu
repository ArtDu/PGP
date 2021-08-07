#include <bits/stdc++.h>
#include <mpi.h>
#include <omp.h>

using namespace std;



int main(int argc, char *argv[]) {
    int numproc, id, device_count = 3;
    cin >> device_count;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    cout << numproc << " " << id << " " << device_count << endl;
    return 0;
}