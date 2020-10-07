#include <stdio.h>
#include <stdlib.h>

void f(double *res, double *vec1, double *vec2, int n) {
  int i = 0;
  for(i = 0; i < n; i++)  
    res[i] = vec1[i] + vec2[i];
}

int main() {
  int i, n;
  scanf("%d", &n); 
  double *res = (double *)malloc(sizeof(double) * n);
  double *vec1 = (double *)malloc(sizeof(double) * n);
  double *vec2 = (double *)malloc(sizeof(double) * n);
  for(i = 0; i < n; i++)
    scanf("%lf", &vec1[i]); 
  for(i = 0; i < n; i++)
    scanf("%lf", &vec2[i]);

  


  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  f(res, vec1, vec2, n);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  fprintf(stderr, "time = %f\n", time);
  cudaEventDestroy(stop);
  cudaEventDestroy(start);


  
  
  // for(i = 0; i < n; i++)
  //  printf("%f ", res[i]);
  // printf("\n");
  free(res);
  free(vec1);
  free(vec2);
  return 0;
}
