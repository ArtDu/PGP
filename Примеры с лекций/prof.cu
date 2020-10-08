#include <stdio.h>

// nvcc prof.cu
// nvprof --query-events
// nvprof --query-metrics
// nvprof -e divergent_branch,global_store_transaction,l1_shared_bank_conflict,l1_local_load_hit -m sm_efficiency ./a.out
//
// divergent_branch - дивергенция нитей
// global_store_transaction - кол-во транзакций к глобальной памяти
// l1_shared_bank_conflict - кол-во конфликтов банков памяти при работе с разделяемой памятью
// l1_local_load_hit - перенос переменных из регистровой памяти в локальную
// sm_efficiency - загрузка мультипроцессоров

__global__ void kernel(float *src, float *dst, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < n && idy < n) 
		dst[idx * n + idy] = src[idy * n + idx];
}

__global__ void kernel_shared(float *src, float *dst, int n) {
	__shared__ float buff[32][32 + 1];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < n && idy < n) 
		buff[threadIdx.x][threadIdx.y] = src[idy * n + idx];
	__syncthreads();
	idx = blockIdx.x * blockDim.x + threadIdx.y;
	idy = blockIdx.y * blockDim.y + threadIdx.x;
	if (idx < n && idy < n)
		dst[idx * n + idy] = buff[threadIdx.y][threadIdx.x];
}

int main() {
	int i, n = 1024;
	float *src = (float *)malloc(sizeof(float) * n * n);
	float *dst = (float *)malloc(sizeof(float) * n * n);
	for(i = 0; i < n * n; i++)
		src[i] = i;
	float *dev_src, *dev_dst;
	cudaMalloc(&dev_src, sizeof(float) * n * n);
	cudaMalloc(&dev_dst, sizeof(float) * n * n);
	cudaMemcpy(dev_src, src, sizeof(float) * n * n, cudaMemcpyHostToDevice);
	
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	kernel_shared<<< dim3(32, 32), dim3(32, 32) >>>(dev_src, dev_dst, n);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	fprintf(stderr, "time = %f\n", time);
	cudaEventDestroy(stop);
	cudaEventDestroy(start);

	cudaMemcpy(dst, dev_dst, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
	cudaFree(dev_src);
	cudaFree(dev_dst);	

	for(i = 0; i < n * n; i++)
		if (src[i] != dst[(i % n) * n + (i / n)]) {	
			fprintf(stderr, "Error\n");
			break;
		}
	free(src);
	free(dst);
	return 0;
}
