#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

// текстурная ссылка <тип элементов, размерность, режим нормализации>
texture<uchar4, 2, cudaReadModeElementType> tex;

__global__ void kernel(uchar4 *out, int w, int h, int wScale, int hScale) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int n = wScale * hScale;
	int x, y, i, j;
	uchar4 p;
	uint4 s;


	for(y = idy; y < h; y += offsety) {
		for(x = idx; x < w; x += offsetx) {

      s = {0,0,0,0};
      
			for (i = 0; i < wScale; ++i) {
				for (j = 0; j < hScale; ++j){
					p = tex2D(tex, x * wScale + i, y * hScale + j);
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
	fread(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	// Подготовка данных для текстуры
	cudaArray *arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	CSC(cudaMallocArray(&arr, &ch, w, h));
	CSC(cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

	// Подготовка текстурной ссылки, настройка интерфейса работы с данными
	tex.addressMode[0] = cudaAddressModeClamp;	// Политика обработки выхода за границы по каждому измерению
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.channelDesc = ch;
	tex.filterMode = cudaFilterModePoint;		// Без интерполяции при обращении по дробным координатам
	tex.normalized = false;						// Режим нормализации координат: без нормализации

	// Связываем интерфейс с данными
	CSC(cudaBindTextureToArray(tex, arr, ch));

	uchar4 *dev_out;
	CSC(cudaMalloc(&dev_out, sizeof(uchar4) * wn * hn));

	kernel<<<dim3(16, 16), dim3(16, 16)>>>(dev_out, wn, hn, wScale, hScale);
	CSC(cudaGetLastError());

	CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * wn * hn, cudaMemcpyDeviceToHost));

	// Отвязываем данные от текстурной ссылки
	CSC(cudaUnbindTexture(tex));

	CSC(cudaFreeArray(arr));
	CSC(cudaFree(dev_out));

	fp = fopen(outputFile, "wb");
	fwrite(&wn, sizeof(int), 1, fp);
	fwrite(&hn, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), wn * hn, fp);
	fclose(fp);

	free(data);
	return 0;
}
