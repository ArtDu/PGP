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



int main() {


	int w, h;
	char inputFile[256], outputFile[256];
	scanf("%s %s %d %d", inputFile, outputFile);

	FILE *fp = fopen(inputFile, "rb");
	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);

	uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
	fread(data, sizeof(uchar4), w * h, fp);
	fclose(fp);


	fp = fopen(outputFile, "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	free(data);
	return 0;
}
