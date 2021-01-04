#include <bits/stdc++.h>
#include <stdlib.h>
#include "mpi.h"

using namespace std;

int main(int argc, char *argv[]) {
    int rank, nprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    fprintf(stderr, "%d(%d)\n", rank, nprocs);
    fflush(stderr);

    char fn[1024];
    cin >> fn;

    MPI_File fp;
    MPI_Bcast(fn, 1024, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_File_delete(fn, MPI_INFO_NULL);
    MPI_File_open(MPI_COMM_WORLD, fn, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fp);

    char out_buff[5] = "1234";
    MPI_File_write_all(fp, out_buff, 1, MPI_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&fp);

    MPI_Finalize();


    return 0;
}
