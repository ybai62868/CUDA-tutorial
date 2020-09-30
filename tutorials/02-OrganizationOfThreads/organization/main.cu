#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>


__global__ void print_threadIds() {
    printf("blockIdx.x : %d, blockIdx.y : %d, blockIdx.z : %d , blockDim.x : %d, blockDim.y : %d, gridDim.x : %d, gridDim.y : %d \n",
            blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y,
            gridDim.x, gridDim.y);

    
}


int main(void)
{
    int nx, ny;
    nx = 16;
    ny = 16;
    dim3 block(8, 8);
    dim3 grid(nx / block.x, ny / block.y);

    print_threadIds <<<grid, block>>> ();
    cudaDeviceSynchronize();

    cudaDeviceReset();


    return 0;
}

