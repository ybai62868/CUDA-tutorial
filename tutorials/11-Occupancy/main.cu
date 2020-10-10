// nvcc -Xptxas=-v -o a.out main.cu 
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <iostream>

__global__ void occupancy_test(int* results)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int x1 = 1;
    int x2 = 2;
    int x3 = 3;
    int x4 = 4;
    int x5 = 5;
    int x6 = 6;
    int x7 = 7;
    int x8 = 8;
    results[gid] = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8;
}

int main(void)
{
     
    
    
    return 0;
}
