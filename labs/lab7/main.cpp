#include <bits/stdc++.h>
#include <algorithm>
#include <mpi/mpi.h>

// Индексация внутри блока
// из 3d табличного представления в 1d одиночный массив
#define _i(i, j, k) (((k) + 1) * (ny + 2) * (nx + 2) + ((j) + 1) * (nx + 2) + (i) + 1)

// Индексация по блокам (процессам)
#define _ib(i, j, k) ((k) * nby * nbx + (j) * nbx + (i))

#define _ibz(id) ((id) / nby / nbx)
#define _iby(id) (((id) % (nby * nbx)) / nbx)
#define _ibx(id) ((id) % nbx)

int main(int argc, char *argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);

    int id, ib, jb, kb, nbx, nby, nbz, nx, ny, nz;
    int i, j, k;
    int numproc, proc_name_len;
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    double lx, ly, lz, hx, hy, hz, bc_down, bc_up, bc_left, bc_right, bc_front, bc_back, eps, diff, u_0;
    double *data, *temp, *next, *buff;
    std::string file_name;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Get_processor_name(proc_name, &proc_name_len);


    if (id == 0) {          // Инициализация параметров расчета
        std::cin >> nbx >> nby >> nbz;          // Размер сетки блоков (процессов)
        std::cin >> nx >> ny >> nz; // Размер блока
        std::cin >> file_name;
        std::cin >> eps;
        std::cin >> lx >> ly >> lz;
        std::cin >> bc_down >> bc_up >> bc_left >> bc_right >> bc_front >> bc_back;
        std::cin >> u_0;
    }

    MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);      // Передача параметров расчета всем процессам
    MPI_Bcast(&ny, 1, MPI_INT, 0, MPI_COMM_WORLD);      // Передача параметров расчета всем процессам
    MPI_Bcast(&nz, 1, MPI_INT, 0, MPI_COMM_WORLD);      // Передача параметров расчета всем процессам
    MPI_Bcast(&nbx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nby, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nbz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_down, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_up, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_left, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_right, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_front, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_back, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    ib = _ibx(id);    // Переход к 3-мерной индексации процессов
    jb = _iby(id);
    kb = _ibz(id);

    hx = lx / (nx * nbx);
    hy = ly / (ny * nby);
    hz = lz / (nz * nbz);

    data = (double *) malloc(sizeof(double) * (nx + 2) * (ny + 2) * (nz + 2));
    next = (double *) malloc(sizeof(double) * (nx + 2) * (ny + 2) * (nz + 2));
    buff = (double *) malloc(sizeof(double) *
                             (std::max(nx, std::max(ny, nz)) * std::max(nx, std::max(ny, nz)) + 2));


    for (i = 0; i < nx; i++)          // Инициализация блока
        for (j = 0; j < ny; j++)
            for (k = 0; k < nz; k++)
                data[_i(i, j, k)] = u_0;

    for (; true;) {
        diff = 0;

// Отправка данных и прием в перемешку

        if (ib + 1 < nbx) {
            for (k = 0; k < nz; ++k)
                for (j = 0; j < ny; j++)
                    buff[k * ny + j] = data[_i(nx - 1, j, k)];
            MPI_Send(buff, ny * nz, MPI_DOUBLE, _ib(ib + 1, jb, kb), id, MPI_COMM_WORLD);
        }

        if (ib > 0) {
            MPI_Recv(buff, ny * nz, MPI_DOUBLE, _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb),
                     MPI_COMM_WORLD, &status);
            for (k = 0; k < nz; ++k)
                for (j = 0; j < ny; j++)
                    data[_i(-1, j, k)] = buff[k * ny + j];
        } else {
            for (k = 0; k < nz; ++k)
                for (j = 0; j < ny; j++)
                    data[_i(-1, j, k)] = bc_left;
        }

        if (jb + 1 < nby) {
            for (k = 0; k < nz; ++k)
                for (i = 0; i < nx; i++)
                    buff[k * nx + i] = data[_i(i, ny - 1, k)];
            MPI_Send(buff, nz * nx, MPI_DOUBLE, _ib(ib, jb + 1, kb), id, MPI_COMM_WORLD);
        }

        if (jb > 0) {
            MPI_Recv(buff, nx * nz, MPI_DOUBLE, _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb),
                     MPI_COMM_WORLD, &status);
            for (k = 0; k < nz; ++k)
                for (i = 0; i < nx; i++)
                    data[_i(i, -1, k)] = buff[k * nx + i];
        } else {
            for (k = 0; k < nz; ++k)
                for (i = 0; i < nx; i++)
                    data[_i(i, -1, k)] = bc_front;
        }

        if (kb + 1 < nbz) {
            for (j = 0; j < ny; ++j)
                for (i = 0; i < nx; i++)
                    buff[j * nx + i] = data[_i(i, j, nz - 1)];
            MPI_Send(buff, ny * nx, MPI_DOUBLE, _ib(ib, jb, kb + 1), id, MPI_COMM_WORLD);
        }

        if (kb > 0) {
            MPI_Recv(buff, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1),
                     MPI_COMM_WORLD, &status);
            for (j = 0; j < ny; ++j)
                for (i = 0; i < nx; i++)
                    data[_i(i, j, -1)] = buff[j * nx + i];
        } else {
            for (j = 0; j < ny; ++j)
                for (i = 0; i < nx; ++i)
                    data[_i(i, j, -1)] = bc_down;
        }

        if (ib > 0) {
            for (k = 0; k < nz; ++k)
                for (j = 0; j < ny; j++)
                    buff[k * ny + j] = data[_i(0, j, k)];
            MPI_Send(buff, nz * ny, MPI_DOUBLE, _ib(ib - 1, jb, kb), id, MPI_COMM_WORLD);
        }

        if (ib + 1 < nbx) {
            MPI_Recv(buff, ny * nz, MPI_DOUBLE, _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb),
                     MPI_COMM_WORLD, &status);
            for (k = 0; k < nz; ++k)
                for (j = 0; j < ny; j++)
                    data[_i(nx, j, k)] = buff[k * ny + j];
        } else {
            for (k = 0; k < nz; ++k)
                for (j = 0; j < ny; ++j)
                    data[_i(nx, j, k)] = bc_right;
        }



        if (jb > 0) {
            for (k = 0; k < nz; ++k)
                for (i = 0; i < nx; i++)
                    buff[k * nx + i] = data[_i(i, 0, k)];
            MPI_Send(buff, nz * nx, MPI_DOUBLE, _ib(ib, jb - 1, kb), id, MPI_COMM_WORLD);
        }

        if (jb + 1 < nby) {
            MPI_Recv(buff, nx * nz, MPI_DOUBLE, _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb),
                     MPI_COMM_WORLD, &status);
            for (k = 0; k < nz; k++)
                for (i = 0; i < nx; i++)
                    data[_i(i, ny, k)] = buff[k * nx + i];
        } else {
            for (k = 0; k < nz; k++)
                for (i = 0; i < nx; i++)
                    data[_i(i, ny, k)] = bc_back;
        }

        if (kb > 0) {
            for (j = 0; j < ny; ++j)
                for (i = 0; i < nx; i++)
                    buff[j * nx + i] = data[_i(i, j, 0)];
            MPI_Send(buff, ny * nx, MPI_DOUBLE, _ib(ib, jb, kb - 1), id, MPI_COMM_WORLD);
        }

        if (kb + 1 < nbz) {
            MPI_Recv(buff, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1),
                     MPI_COMM_WORLD, &status);
            for (j = 0; j < ny; ++j)
                for (i = 0; i < nx; i++)
                    data[_i(i, j, nz)] = buff[i + j * nx];
        } else {
            for (j = 0; j < ny; ++j)
                for (i = 0; i < nx; ++i)
                    data[_i(i, j, nz)] = bc_up;
        }

        
//		Перевычисление значений температуры


        for (i = 0; i < nx; i++)
            for (j = 0; j < ny; j++)
                for (k = 0; k < nz; k++) {
                    next[_i(i, j, k)] = 0.5 * ((data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (hx * hx) +
                                               (data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (hy * hy) +
                                               (data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (hz * hz)) /
                                        (1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));
                    diff = std::max(diff, fabs(next[_i(i, j, k)] - data[_i(i, j, k)]));
                }


        temp = next;
        next = data;
        data = temp;


        double *diffs = (double *) malloc(sizeof(double) * nbx * nby * nbz);
        MPI_Allgather(&diff, 1, MPI_DOUBLE, diffs, 1, MPI_DOUBLE,
                      MPI_COMM_WORLD);
        double gather_diff = 0;
        for (k = 0; k < nbx * nby * nbz; ++k) {
            gather_diff = std::max(gather_diff, diffs[k]);
        }

        if (gather_diff < eps) {
            break;
        }


    }
    if (id != 0) {
        for (k = 0; k < nz; ++k)
            for (j = 0; j < ny; j++) {
                for (i = 0; i < nx; i++)
                    buff[i] = data[_i(i, j, k)];
                MPI_Send(buff, nx, MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
            }
    } else {
        FILE *fd;
        fd = fopen(file_name.c_str(), "w");
        for (kb = 0; kb < nbz; kb++)
            for (k = 0; k < nz; ++k)
                for (jb = 0; jb < nby; jb++)
                    for (j = 0; j < ny; j++)
                        for (ib = 0; ib < nbx; ib++) {
                            if (_ib(ib, jb, kb) == 0)
                                for (i = 0; i < nx; i++)
                                    buff[i] = data[_i(i, j, k)];
                            else
                                MPI_Recv(buff, nx, MPI_DOUBLE, _ib(ib, jb, kb), _ib(ib, jb, kb), MPI_COMM_WORLD,
                                         &status);
                            for (i = 0; i < nx; i++) {
//                                printf("%.2f ", buff[i]);
                                fprintf(fd, "%.7e ", buff[i]);
                            }
                        }
        fclose(fd);
    }

    MPI_Finalize();

    free(buff);
    free(data);
    free(next);


    return 0;
}

