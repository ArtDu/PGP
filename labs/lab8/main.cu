#include <bits/stdc++.h>
#include <algorithm>
#include <mpi.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#define CSC(call)                           \
do {                                \
  cudaError_t res = call;                     \
  if (res != cudaSuccess) {                   \
    fprintf(stderr, "ERROR in %s:%d. Message: %s\n",      \
        __FILE__, __LINE__, cudaGetErrorString(res));   \
    exit(0);                          \
  }                               \
} while(0)


// Индексация внутри блока
// из 3d табличного представления в 1d одиночный массив
#define _i(i, j, k) (((k) + 1) * (ny + 2) * (nx + 2) + ((j) + 1) * (nx + 2) + (i) + 1)
#define _ixy(i, j) ((j) * nx + (i))
#define _ixz(i, k) ((k) * nx + (i))
#define _iyz(j, k) ((k) * ny + (j))

// Индексация по блокам (процессам)
#define _ib(i, j, k) ((k) * nby * nbx + (j) * nbx + (i))


#define _ibz(id) ((id) / nby / nbx)
#define _iby(id) (((id) % (nby * nbx)) / nbx)
#define _ibx(id) ((id) % nbx)

__global__ void kernel_copy_yz(double *plane_yz, double *data, int nx, int ny, int nz, int i, int dir, int bc) {
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsety = blockDim.y * gridDim.y;
    int offsetx = blockDim.x * gridDim.x;
    int j, k;
    if (dir) {
        for (k = idy; k < nz; k += offsety)
            for (j = idx; j < ny; j += offsetx)
                plane_yz[_iyz(j, k)] = data[_i(i, j, k)];
    } else {
        if (plane_yz) {
            for (k = idy; k < nz; k += offsety)
                for (j = idx; j < ny; j += offsetx)
                    data[_i(i, j, k)] = plane_yz[_iyz(j, k)];
        } else {
            for (k = idy; k < nz; k += offsety)
                for (j = idx; j < ny; j += offsetx)
                    data[_i(i, j, k)] = bc;
        }
    }
}

__global__ void kernel_copy_xz(double *plane_xz, double *data, int nx, int ny, int nz, int j, int dir, int bc) {
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsety = blockDim.y * gridDim.y;
    int offsetx = blockDim.x * gridDim.x;
    int i, k;
    if (dir) {
        for (k = idy; k < nz; k += offsety)
            for (i = idx; i < nx; i += offsetx)
                plane_xz[_ixz(i, k)] = data[_i(i, j, k)];
    } else {
        if (plane_xz) {
            for (k = idy; k < nz; k += offsety)
                for (i = idx; i < nx; i += offsetx)
                    data[_i(i, j, k)] = plane_xz[_ixz(i, k)];
        } else {
            for (k = idy; k < nz; k += offsety)
                for (i = idx; i < nx; i += offsetx)
                    data[_i(i, j, k)] = bc;
        }
    }
}

__global__ void kernel_copy_xy(double *plane_xy, double *data, int nx, int ny, int nz, int k, int dir, int bc) {
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsety = blockDim.y * gridDim.y;
    int offsetx = blockDim.x * gridDim.x;
    int i, j;
    if (dir) {
        for (j = idy; j < ny; j += offsety)
            for (i = idx; i < nx; i += offsetx)
                plane_xy[_ixy(i, j)] = data[_i(i, j, k)];
    } else {
        if (plane_xy) {
            for (j = idy; j < ny; j += offsety)
                for (i = idx; i < nx; i += offsetx)
                    data[_i(i, j, k)] = plane_xy[_ixy(i, j)];
        } else {
            for (j = idy; j < ny; j += offsety)
                for (i = idx; i < nx; i += offsetx)
                    data[_i(i, j, k)] = bc;
        }
    }
}

__global__ void kernel(double *next, double *data, int nx, int ny, int nz, double hx, double hy, double hz) {
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetz = blockDim.z * gridDim.z;
    int offsety = blockDim.y * gridDim.y;
    int offsetx = blockDim.x * gridDim.x;
    int i, j, k;
    for (i = idx; i < nx; i += offsetx)
        for (j = idy; j < ny; j += offsety)
            for (k = idz; k < nz; k += offsetz) {
                next[_i(i, j, k)] = 0.5 * ((data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (hx * hx) +
                                           (data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (hy * hy) +
                                           (data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (hz * hz)) /
                                    (1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));
            }
}

__global__ void kernel_error(double *next, double *data, int nx, int ny, int nz) {
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetz = blockDim.z * gridDim.z;
    int offsety = blockDim.y * gridDim.y;
    int offsetx = blockDim.x * gridDim.x;
    int i, j, k;
    for (i = idx - 1; i < nx + 1; i += offsetx)
        for (j = idy - 1; j < ny + 1; j += offsety)
            for (k = idz - 1; k < nz + 1; k += offsetz) {
                data[_i(i, j, k)] = ((i != -1 && j != -1 && k != -1 && i != nx && j != ny && k != nz))
                                    * fabs(next[_i(i, j, k)] - data[_i(i, j, k)]);
            }
}

int main(int argc, char *argv[]) {
    std::ios_base::sync_with_stdio(false);
//    std::cin.tie(nullptr);
//    std::cout.tie(nullptr);

    int id, ib, jb, kb, nbx, nby, nbz, nx, ny, nz, it = 0;
    int i, j, k;
    int numproc, proc_name_len;
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    double lx, ly, lz, hx, hy, hz, bc_down, bc_up, bc_left, bc_right, bc_front, bc_back, eps, diff, u_0;
    double *data, *temp, *next, *buff;
    double *dev_data, *dev_next, *dev_buff;

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

    int _size_b = (nx + 2) * (ny + 2) * (nz + 2);
    int _size_plane = (std::max(nx, std::max(ny, nz)) * std::max(nx, std::max(ny, nz)) + 2);
    data = (double *) malloc(sizeof(double) * _size_b);
    double *_data = (double *) malloc(sizeof(double) * _size_b);
    next = (double *) malloc(sizeof(double) * _size_b);
    buff = (double *) malloc(sizeof(double) * _size_plane);

    CSC(cudaMalloc(&dev_data, sizeof(double) * _size_b));
    CSC(cudaMalloc(&dev_next, sizeof(double) * _size_b));
    CSC(cudaMalloc(&dev_buff, sizeof(double) * _size_plane));


    for (i = 0; i < nx; i++)          // Инициализация блока
        for (j = 0; j < ny; j++)
            for (k = 0; k < nz; k++)
                data[_i(i, j, k)] = u_0;

    CSC(cudaMemcpy(dev_data, data, sizeof(double) * _size_b, cudaMemcpyHostToDevice));

    dim3 blocks(32, 32);
    dim3 threads(32, 32);

    for (; true; it++) {
        diff = 0;

// Отправка данных и прием в перемешку

        if (ib + 1 < nbx) {
            kernel_copy_yz<<<blocks, threads>>> (dev_buff, dev_data, nx, ny, nz, nx - 1, true, 0.0);
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(buff, dev_buff, sizeof(double) * _size_plane, cudaMemcpyDeviceToHost));
//            for (k = 0; k < nz; ++k)
//                for (j = 0; j < ny; j++) {
//                    std::cerr << buff[_iyz(j, k)] << " " << data[_i(nx - 1, j, k)] << "\n";
//                    buff[_iyz(j, k)] = data[_i(nx - 1, j, k)];
//                }
            MPI_Send(buff, ny * nz, MPI_DOUBLE, _ib(ib + 1, jb, kb), id, MPI_COMM_WORLD);
        }

        if (ib > 0) {
            MPI_Recv(buff, ny * nz, MPI_DOUBLE, _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb),
                     MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(dev_buff, buff, sizeof(double) * _size_plane, cudaMemcpyHostToDevice));
            kernel_copy_yz<<<blocks, threads>>> (dev_buff, dev_data, nx, ny, nz, -1, false, 0.0);
//            CSC(cudaMemcpy(_data, dev_data, sizeof(double) * _size_b, cudaMemcpyDeviceToHost));
//            for (k = 0; k < nz; ++k)
//                for (j = 0; j < ny; j++) {
//                    std::cerr << _data[_i(-1, j, k)] << " " << buff[_iyz(j, k)] << "\n";
//                    data[_i(-1, j, k)] = buff[_iyz(j, k)];
//                }
        } else {
            kernel_copy_yz<<<blocks, threads>>> (NULL, dev_data, nx, ny, nz, -1, false, bc_left);
//            CSC(cudaMemcpy(_data, dev_data, sizeof(double) * _size_b, cudaMemcpyDeviceToHost));
//            for (k = 0; k < nz; ++k)
//                for (j = 0; j < ny; j++) {
//                    if(_data[_i(-1, j, k)] != bc_left)
//                        std::cerr << "bc_left: " << _data[_i(-1, j, k)] << " " << bc_left << "\n";
//                    data[_i(-1, j, k)] = bc_left;
//                }
        }

        if (jb + 1 < nby) {
            kernel_copy_xz<<<blocks, threads>>> (dev_buff, dev_data, nx, ny, nz, ny - 1, true, 0.0);
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(buff, dev_buff, sizeof(double) * _size_plane, cudaMemcpyDeviceToHost));
//            for (k = 0; k < nz; ++k)
//                for (i = 0; i < nx; i++) {
//                    std::cerr << buff[_ixz(i, k)] << " " << data[_i(i, ny - 1, k)] << "\n";
//                    buff[_ixz(i, k)] = data[_i(i, ny - 1, k)];
//                }
            MPI_Send(buff, nz * nx, MPI_DOUBLE, _ib(ib, jb + 1, kb), id, MPI_COMM_WORLD);
        }

        if (jb > 0) {
            MPI_Recv(buff, nx * nz, MPI_DOUBLE, _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb),
                     MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(dev_buff, buff, sizeof(double) * _size_plane, cudaMemcpyHostToDevice));
            kernel_copy_xz<<<blocks, threads>>> (dev_buff, dev_data, nx, ny, nz, -1, false, 0.0);
//            CSC(cudaMemcpy(_data, dev_data, sizeof(double) * _size_b, cudaMemcpyDeviceToHost));
//            for (k = 0; k < nz; ++k)
//                for (i = 0; i < nx; i++) {
//                    std::cerr << _data[_i(i, -1, k)] << " " << buff[_ixz(i, k)] << "\n";
//                    data[_i(i, -1, k)] = buff[_ixz(i, k)];
//                }
        } else {
            kernel_copy_xz<<<blocks, threads>>> (NULL, dev_data, nx, ny, nz, -1, false, bc_front);
//            CSC(cudaMemcpy(_data, dev_data, sizeof(double) * _size_b, cudaMemcpyDeviceToHost));
//            for (k = 0; k < nz; ++k)
//                for (i = 0; i < nx; i++) {
//                    if(_data[_i(i, -1, k)] != bc_front)
//                        std::cerr << "bc_front: " << _data[_i(i, -1, k)] << " " << bc_front << "\n";
//                    data[_i(i, -1, k)] = bc_front;
//                }
        }

        if (kb + 1 < nbz) {
            kernel_copy_xy<<<blocks, threads>>> (dev_buff, dev_data, nx, ny, nz, nz - 1, true, 0.0);
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(buff, dev_buff, sizeof(double) * _size_plane, cudaMemcpyDeviceToHost));
//            for (j = 0; j < ny; ++j)
//                for (i = 0; i < nx; i++) {
//                    if(buff[_ixy(i, j)] != data[_i(i, j, nz - 1)])
//                        std::cerr << buff[_ixy(i, j)] << " " << data[_i(i, j, nz - 1)] << "\n";
//                    buff[_ixy(i, j)] = data[_i(i, j, nz - 1)];
//                }
            MPI_Send(buff, ny * nx, MPI_DOUBLE, _ib(ib, jb, kb + 1), id, MPI_COMM_WORLD);
        }

        if (kb > 0) {
            MPI_Recv(buff, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1),
                     MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(dev_buff, buff, sizeof(double) * _size_plane, cudaMemcpyHostToDevice));
            kernel_copy_xy<<<blocks, threads>>> (dev_buff, dev_data, nx, ny, nz, -1, false, 0.0);
//            CSC(cudaMemcpy(_data, dev_data, sizeof(double) * _size_b, cudaMemcpyDeviceToHost));
//            for (j = 0; j < ny; ++j)
//                for (i = 0; i < nx; i++) {
//                    if(_data[_i(i, j, -1)] != buff[_ixy(i, j)]) {
//                        std::cerr << _data[_i(i, j, -1)] << " " << buff[_ixy(i, j)] << "\n";
//                    }
//                    data[_i(i, j, -1)] = buff[_ixy(i, j)];
//                }
        } else {
            kernel_copy_xy<<<blocks, threads>>> (NULL, dev_data, nx, ny, nz, -1, false, bc_down);
//            CSC(cudaMemcpy(_data, dev_data, sizeof(double) * _size_b, cudaMemcpyDeviceToHost));
//            for (j = 0; j < ny; ++j)
//                for (i = 0; i < nx; ++i) {
//                    if(_data[_i(i, j, -1)] != bc_down) {
//                        std::cerr << _data[_i(i, j, -1)] << " " << bc_down << "\n";
//                    }
//                    data[_i(i, j, -1)] = bc_down;
//                }
        }

        if (ib > 0) {
            kernel_copy_yz<<<blocks, threads>>> (dev_buff, dev_data, nx, ny, nz, 0, true, 0.0);
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(buff, dev_buff, sizeof(double) * _size_plane, cudaMemcpyDeviceToHost));
//            for (k = 0; k < nz; ++k)
//                for (j = 0; j < ny; j++) {
//                    if(buff[_iyz(j, k)] != data[_i(0, j, k)])
//                        std::cerr << buff[_iyz(j, k)] << " " << data[_i(0, j, k)] << "\n";
//                    buff[_iyz(j, k)] = data[_i(0, j, k)];
//                }
            MPI_Send(buff, nz * ny, MPI_DOUBLE, _ib(ib - 1, jb, kb), id, MPI_COMM_WORLD);
        }

        if (ib + 1 < nbx) {
            MPI_Recv(buff, ny * nz, MPI_DOUBLE, _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb),
                     MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(dev_buff, buff, sizeof(double) * _size_plane, cudaMemcpyHostToDevice));
            kernel_copy_yz<<<blocks, threads>>> (dev_buff, dev_data, nx, ny, nz, nx, false, 0.0);
//            CSC(cudaMemcpy(_data, dev_data, sizeof(double) * _size_b, cudaMemcpyDeviceToHost));
//            for (k = 0; k < nz; ++k)
//                for (j = 0; j < ny; j++) {
//                    if (_data[_i(nx, j, k)] != buff[_iyz(j, k)]) {
//                        std::cerr << _data[_i(nx, j, k)] << " " << buff[_iyz(j, k)] << "\n";
//                    }
//                    data[_i(nx, j, k)] = buff[_iyz(j, k)];
//                }
        } else {
            kernel_copy_yz<<<blocks, threads>>> (NULL, dev_data, nx, ny, nz, nx, false, bc_right);
//            CSC(cudaMemcpy(_data, dev_data, sizeof(double) * _size_b, cudaMemcpyDeviceToHost));
//            for (k = 0; k < nz; ++k)
//                for (j = 0; j < ny; ++j) {
//                    if (_data[_i(nx, j, k)] != bc_right) {
//                        std::cerr << _data[_i(nx, j, k)] << " " << bc_right << "\n";
//                    }
//                    data[_i(nx, j, k)] = bc_right;
//                }
        }


        if (jb > 0) {
            kernel_copy_xz<<<blocks, threads>>> (dev_buff, dev_data, nx, ny, nz, 0, true, 0.0);
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(buff, dev_buff, sizeof(double) * _size_plane, cudaMemcpyDeviceToHost));
//            for (k = 0; k < nz; ++k)
//                for (i = 0; i < nx; i++) {
//                    if (buff[_ixz(i, k)] != data[_i(i, 0, k)]) {
//                        std::cerr << buff[_ixz(i, k)] << " " << data[_i(i, 0, k)] << "\n";
//                    }
//                    buff[_ixz(i, k)] = data[_i(i, 0, k)];
//                }
            MPI_Send(buff, nz * nx, MPI_DOUBLE, _ib(ib, jb - 1, kb), id, MPI_COMM_WORLD);
        }

        if (jb + 1 < nby) {
            MPI_Recv(buff, nx * nz, MPI_DOUBLE, _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb),
                     MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(dev_buff, buff, sizeof(double) * _size_plane, cudaMemcpyHostToDevice));
            kernel_copy_xz<<<blocks, threads>>> (dev_buff, dev_data, nx, ny, nz, ny, false, 0.0);
//            CSC(cudaMemcpy(_data, dev_data, sizeof(double) * _size_b, cudaMemcpyDeviceToHost));
//            for (k = 0; k < nz; k++)
//                for (i = 0; i < nx; i++) {
//                    if (_data[_i(i, ny, k)] != buff[_ixz(i, k)]) {
//                        std::cerr << _data[_i(i, ny, k)] << " " << buff[_ixz(i, k)] << "\n";
//                    }
//                    data[_i(i, ny, k)] = buff[_ixz(i, k)];
//                }
        } else {
            kernel_copy_xz<<<blocks, threads>>> (NULL, dev_data, nx, ny, nz, ny, false, bc_back);
//            CSC(cudaMemcpy(_data, dev_data, sizeof(double) * _size_b, cudaMemcpyDeviceToHost));
//            for (k = 0; k < nz; k++)
//                for (i = 0; i < nx; i++) {
//                    if (_data[_i(i, ny, k)] != bc_back) {
//                        std::cerr << _data[_i(i, ny, k)] << " " << bc_back << "\n";
//                    }
//                    data[_i(i, ny, k)] = bc_back;
//                }
        }

        if (kb > 0) {
            kernel_copy_xy<<<blocks, threads>>> (dev_buff, dev_data, nx, ny, nz, 0, true, 0.0);
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(buff, dev_buff, sizeof(double) * _size_plane, cudaMemcpyDeviceToHost));
//            for (j = 0; j < ny; ++j)
//                for (i = 0; i < nx; i++) {
//                    if(buff[_ixy(i, j)] != data[_i(i, j, 0)]) {
//                        std::cerr << buff[_ixy(i, j)] << " " << data[_i(i, j, 0)] << "\n";
//                    }
//                    buff[_ixy(i, j)] = data[_i(i, j, 0)];
//                }
            MPI_Send(buff, ny * nx, MPI_DOUBLE, _ib(ib, jb, kb - 1), id, MPI_COMM_WORLD);
        }

        if (kb + 1 < nbz) {
            MPI_Recv(buff, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1),
                     MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(dev_buff, buff, sizeof(double) * _size_plane, cudaMemcpyHostToDevice));
            kernel_copy_xy<<<blocks, threads>>> (dev_buff, dev_data, nx, ny, nz, nz, false, 0.0);
//            CSC(cudaMemcpy(_data, dev_data, sizeof(double) * _size_b, cudaMemcpyDeviceToHost));
//            for (j = 0; j < ny; ++j)
//                for (i = 0; i < nx; i++) {
//                    if(_data[_i(i, j, nz)] != buff[_ixy(i, j)]) {
//                        std::cerr << _data[_i(i, j, nz)] << " " << buff[_ixy(i, j)] << "\n";
//                    }
//                    data[_i(i, j, nz)] = buff[_ixy(i, j)];
//                }
        } else {
            kernel_copy_xy<<<blocks, threads>>> (NULL, dev_data, nx, ny, nz, nz, false, bc_up);
//            CSC(cudaMemcpy(_data, dev_data, sizeof(double) * _size_b, cudaMemcpyDeviceToHost));
//            for (j = 0; j < ny; ++j)
//                for (i = 0; i < nx; ++i) {
//                    if(_data[_i(i, j, nz)] != bc_up) {
//                        std::cerr << "bc_up: " << _data[_i(i, j, nz)] << " " << bc_up << "\n";
//                    }
//                    data[_i(i, j, nz)] = bc_up;
//                }
        }


//      Перевычисление значений температуры
//        CSC(cudaMemcpy(dev_data, data, sizeof(double) * _size_b, cudaMemcpyHostToDevice));
        kernel<<<dim3(8,8,8), dim3(32, 4, 4)>>>(dev_next, dev_data, nx, ny, nz, hx, hy, hz);
        CSC(cudaGetLastError());
        kernel_error<<<dim3(8,8,8), dim3(32, 4, 4)>>> (dev_next, dev_data, nx, ny, nz);
        CSC(cudaGetLastError());


        double g_error, error = 0.0;
        thrust::device_ptr< double > p_arr = thrust::device_pointer_cast(dev_data);
        thrust::device_ptr< double > res = thrust::max_element(p_arr, p_arr + _size_b);
        error = *res;

        temp = dev_data;
        dev_data = dev_next;
        dev_next = temp;


//        CSC(cudaMemcpy(data, dev_data, sizeof(double) * _size_b, cudaMemcpyDeviceToHost));
//        for (i = 0; i < nx; i++)
//            for (j = 0; j < ny; j++)
//                for (k = 0; k < nz; k++) {
//                    next[_i(i, j, k)] = 0.5 * ((data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (hx * hx) +
//                                               (data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (hy * hy) +
//                                               (data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (hz * hz)) /
//                                        (1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));
//                    diff = std::max(diff, fabs(next[_i(i, j, k)] - data[_i(i, j, k)]));
//                }
//
//
//        temp = next;
//        next = data;
//        data = temp;


        double *diffs = (double *) malloc(sizeof(double) * nbx * nby * nbz);
        MPI_Allgather(&error, 1, MPI_DOUBLE, diffs, 1, MPI_DOUBLE,
                      MPI_COMM_WORLD);
        double gather_diff = 0;
        for (k = 0; k < nbx * nby * nbz; ++k) {
            gather_diff = std::max(gather_diff, diffs[k]);
        }

        if (gather_diff < eps) {
            break;
        }
//        std::cerr << gather_diff << " ";
//
//        diffs = (double *) malloc(sizeof(double) * nbx * nby * nbz);
//        MPI_Allgather(&diff, 1, MPI_DOUBLE, diffs, 1, MPI_DOUBLE,
//                      MPI_COMM_WORLD);
//        gather_diff = 0;
//        for (k = 0; k < nbx * nby * nbz; ++k) {
//            gather_diff = std::max(gather_diff, diffs[k]);
//        }
//        std::cerr << gather_diff << "\n";

    }
    CSC(cudaMemcpy(data, dev_data, sizeof(double) * _size_b, cudaMemcpyDeviceToHost));
    CSC(cudaFree(dev_data));
    CSC(cudaFree(dev_next));
    CSC(cudaFree(dev_buff));

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
    std::cerr <<"it: "<< it << "\n";

    free(buff);
    free(data);
    free(next);


    return 0;
}

