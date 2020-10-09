#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include "cuda_common.cuh"
#include <cstdio>


#include <cstdlib>
#include <ctime>
#include <cstring>


__global__ void sum_array_gpu(int* a, int* b, int* c, int size)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
      c[gid] = a[gid] + b[gid];
    }
}

void sum_array_cpu(int* a, int* b, int* c, int size)
{
    for ( int i = 0;i < size;i++ ) {
        c[i] = a[i] + b[i];      
    }
}


void compare_arrays(int* a, int* b, int size)
{
    for ( int i = 0;i < size;i++ ) {
      if (a[i] != b[i]) { 
          printf("Arrays are different!");
          return;
      }
    }
    printf("Arrays are same\n");
}


int main(void)
{
    int size = 10000;
    int block_size = 256;

    int NO_BYTES = size * sizeof(int);

    // host pointer
    int* h_a, *h_b, *gpu_results;
    int* h_c;
    h_a = (int*)malloc(NO_BYTES);
    h_b = (int*)malloc(NO_BYTES);
    h_c = (int*)malloc(NO_BYTES);
    gpu_results = (int*)malloc(NO_BYTES);
    
  

    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0;i < size;i++) {
        h_a[i] = (int)(rand() & 0xff);
        h_b[i] = (int)(rand() & 0xff);
    }
    
    clock_t cpu_start, cpu_end;
    cpu_start = clock();
    sum_array_cpu(h_a, h_b, h_c, size);
    cpu_end = clock();
    printf("Sum array CPU execution time : %4.6f \n",(double)((double)(cpu_end -
                    cpu_start)/CLOCKS_PER_SEC));
    memset(gpu_results, 0, NO_BYTES);


    // device pointer
    int* d_a, *d_b, *d_c;
    cudaMalloc((int**)&d_a, NO_BYTES);
    cudaMalloc((int**)&d_b, NO_BYTES);
    cudaMalloc((int**)&d_c, NO_BYTES);

    clock_t htod_start, htod_end;
    htod_start = clock();
    cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice);
    htod_end = clock();
    printf("Sum array host to device time : %4.6f \n",(double)((double)(htod_end -
                    htod_start)/CLOCKS_PER_SEC));


    // launching the grid
    dim3 block(block_size);
    dim3 grid((size / block.x) + 1);

    clock_t gpu_start, gpu_end;
    gpu_start = clock();
    sum_array_gpu<<<grid, block>>> (d_a, d_b, d_c, size);
    gpu_end = clock();
    printf("Sum array GPU execution time : %4.6f \n",(double)((double)(gpu_end -
                    gpu_start)/CLOCKS_PER_SEC));
    cudaDeviceSynchronize();

    clock_t dtoh_start, dtoh_end;
    dtoh_start = clock();
    cudaMemcpy(gpu_results, d_c, NO_BYTES, cudaMemcpyDeviceToHost);
    dtoh_end = clock();
    printf("Sum array GPU total time : %4.6f \n",(double)((double)(dtoh_end -
                    htod_start)/CLOCKS_PER_SEC));

    // array comparison
    compare_arrays(h_c, gpu_results, size);



    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    

    free(gpu_results);
    cudaDeviceReset();


    return 0;

}
