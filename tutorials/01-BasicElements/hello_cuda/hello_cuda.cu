#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>


__global__ void hello_cuda() {

	printf("hello CUDA world\n");
}

int main(void)
{
	// hello_cuda<<<1, 1>>>();
	//hello_cuda<<<1, 20>>>();
	// dim3 block(4);
	// dim3 grid(8);
	dim3 block(8, 2);
	dim3 grid(2, 2);

	hello_cuda<<<block, grid>>>();
	
	cudaDeviceSynchronize();
	
	cudaDeviceReset();

	return 0;

}
