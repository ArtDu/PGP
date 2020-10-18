#include <bits/stdc++.h>

using namespace std;

const int MAXX = 1e8;

__constant__ int4 avg_dev[32];
__constant__ double cov_inv_dev[32][3][3];
__constant__ double dets_dev[32];

#define CSC(call)                            \
do {                                \
  cudaError_t res = call;                      \
  if (res != cudaSuccess) {                    \
    fprintf(stderr, "ERROR in %s:%d. Message: %s\n",      \
        __FILE__, __LINE__, cudaGetErrorString(res));    \
    exit(0);                          \
  }                                \
} while(0)


struct pnt {
    int x, y;
};


__global__ void kernel(uchar4 *data, int w, int h, int nc) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int x, y, i, j, k;
    uchar4 ps;

    for (y = idy; y < h; y += offsety) {
        for (x = idx; x < w; x += offsetx) {
            ps = data[y * w + x];

            double mx = -MAXX;
            int idx = -1;
            for (i = 0; i < nc; ++i) {

                int diff[3];
                diff[0] = ps.x - avg_dev[i].x;
                diff[1] = ps.y - avg_dev[i].y;
                diff[2] = ps.z - avg_dev[i].z;

                double tmp[3];
                for (j = 0; j < 3; ++j) {
                    tmp[j] = 0;
                    for (k = 0; k < 3; ++k) {
                        tmp[j] += (diff[k] * cov_inv_dev[i][k][j]);
                    }
                }
                double ans = 0;
                for (j = 0; j < 3; ++j) {
                    ans += (tmp[j] * diff[j]);
                }
                ans = -ans - log(abs(dets_dev[i]));

                if (ans > mx) {
                    mx = ans;
                    idx = i;
                }
            }
            data[y * w + x].w = idx;
        }
    }
}

int main() {


    int w, h;
    char inputFile[256], outputFile[256];
    cin >> inputFile >> outputFile;

    FILE *fp = fopen(inputFile, "rb");
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);

    uchar4 *data = (uchar4 *) malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    int nc, np;
    cin >> nc;
    vector<vector<pnt>> classes(nc);
    int4 avg[32];
    double cov[32][3][3];
    double cov_inv[32][3][3];
    double dets[32];
    for (int i = 0; i < nc; ++i) {
        cin >> np;
        classes[i].resize(np);
        // input + counting averages
        for (int j = 0; j < np; ++j) {
            cin >> classes[i][j].x >> classes[i][j].y;
            uchar4 ps = data[classes[i][j].y * w + classes[i][j].x];
            avg[i].x += ps.x;
            avg[i].y += ps.y;
            avg[i].z += ps.z;
        }
        avg[i].x /= np;
        avg[i].y /= np;
        avg[i].z /= np;

        // counting cov
        for (int j = 0; j < np; ++j) {
            uchar4 ps = data[classes[i][j].y * w + classes[i][j].x];


            int diff[3];
            diff[0] = ps.x - avg[i].x;
            diff[1] = ps.y - avg[i].y;
            diff[2] = ps.z - avg[i].z;

            for (int k = 0; k < 3; ++k) {
                for (int m = 0; m < 3; ++m) {
                    cov[i][k][m] += diff[k] * diff[m];
                }
            }
        }
        for (int k = 0; k < 3; ++k) {
            for (int m = 0; m < 3; ++m) {
                cov[i][k][m] /= (np - 1);
            }
        }

        // counting cov_inverse + determinants
        double det = cov[i][0][0] * (cov[i][1][1] * cov[i][2][2] - cov[i][2][1] * cov[i][1][2])
                     - cov[i][0][1] * (cov[i][1][0] * cov[i][2][2] - cov[i][2][0] * cov[i][1][2])
                     + cov[i][0][2] * (cov[i][1][0] * cov[i][2][1] - cov[i][2][0] * cov[i][1][1]);


        cov_inv[i][0][0] = (cov[i][1][1] * cov[i][2][2] - cov[i][2][1] * cov[i][1][2]) / det;
        cov_inv[i][1][0] = -(cov[i][1][0] * cov[i][2][2] - cov[i][2][0] * cov[i][1][2]) / det;
        cov_inv[i][2][0] = (cov[i][1][0] * cov[i][2][1] - cov[i][2][0] * cov[i][1][1]) / det;

        cov_inv[i][0][1] = -(cov[i][0][1] * cov[i][2][2] - cov[i][2][1] * cov[i][0][2]) / det;
        cov_inv[i][1][1] = (cov[i][0][0] * cov[i][2][2] - cov[i][2][0] * cov[i][0][2]) / det;
        cov_inv[i][2][1] = -(cov[i][0][0] * cov[i][2][1] - cov[i][2][0] * cov[i][0][1]) / det;

        cov_inv[i][0][2] = (cov[i][0][1] * cov[i][1][2] - cov[i][1][1] * cov[i][0][2]) / det;
        cov_inv[i][1][2] = -(cov[i][0][0] * cov[i][1][2] - cov[i][1][0] * cov[i][0][2]) / det;
        cov_inv[i][2][2] = (cov[i][0][0] * cov[i][1][1] - cov[i][1][0] * cov[i][0][1]) / det;

        dets[i] = det;
    }

    uchar4 *dev_data;
    CSC(cudaMalloc(&dev_data, sizeof(uchar4) * w * h));
    CSC(cudaMemcpy(dev_data, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

    CSC(cudaMemcpyToSymbol(avg_dev, avg, sizeof(double) * 32 * 3));
    CSC(cudaMemcpyToSymbol(cov_inv_dev, cov_inv, sizeof(double) * 32 * 3 * 3));
    CSC(cudaMemcpyToSymbol(dets_dev, dets, sizeof(double) * 32));

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    kernel<<<dim3(16, 16), dim3(16, 16)>>>(dev_data, w, h, nc);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    fprintf(stderr, "%.2f\n", time);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    CSC(cudaMemcpy(data, dev_data, sizeof(uchar4) * h * w, cudaMemcpyDeviceToHost));


    fp = fopen(outputFile, "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    cudaFree(dev_data);
    free(data);
    return 0;
}
