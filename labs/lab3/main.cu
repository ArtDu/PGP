#include <bits/stdc++.h>

using namespace std;

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
        // input + count averages
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

        // count cov
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

        // count cov_inverse + determinants
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



//    for (int y = 0; y < h; ++y){
//        for (int x = 0; x < w; ++x){
//            uchar4 ps = data[y * w + x];
//        }
//    }



    fp = fopen(outputFile, "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    return 0;
}
