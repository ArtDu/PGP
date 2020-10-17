#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define CSC(call)                           \
do {                                \
  cudaError_t res = call;                     \
  if (res != cudaSuccess) {                   \
    fprintf(stderr, "ERROR in %s:%d. Message: %s\n",      \
        __FILE__, __LINE__, cudaGetErrorString(res));   \
    exit(0);                          \
  }                               \
} while(0)


void f(uchar4* data, uchar4 *out, int w, int h, int wScale, int hScale) {
  
  int n = wScale * hScale;
  int x, y, i, j;
  uchar4 p;
  uint4 s;


  for(y = 0; y < h; y += 1) {
    for(x = 0; x < w; x += 1) {

      s = {0,0,0,0};
      
      for (i = 0; i < wScale; ++i) {
        for (j = 0; j < hScale; ++j){
          p = data[(x * wScale + i) + y * (y * hScale + j)];
          s.x += p.x;
          s.y += p.y;
          s.z += p.z;
        }
      }
      s.x /= n;
      s.y /= n;
      s.z /= n;
    
      out[y * w + x] = make_uchar4(s.x, s.y, s.z, s.w);
    }
  }
}

int main() {


  int w, h, wn, hn, wScale, hScale;
  char inputFile[256], outputFile[256];
  scanf("%s %s %d %d", inputFile, outputFile, &wn, &hn);

  FILE *fp = fopen(inputFile, "rb");
  fread(&w, sizeof(int), 1, fp);
  fread(&h, sizeof(int), 1, fp);

  wScale = w / wn, hScale = h / hn;

  uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
  uchar4 *out = (uchar4 *)malloc(sizeof(uchar4) * w * h);
  fread(data, sizeof(uchar4), w * h, fp);
  fclose(fp);

  

  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  f(data, out, wn, hn, wScale, hScale);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  fprintf(stderr, "%.2f\n", time);
  cudaEventDestroy(stop);
  cudaEventDestroy(start);

  
  return 0;
}
