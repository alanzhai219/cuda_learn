#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define NB 1000
// increase array length here if your GPU has more than 32 SMs
#define MAX_SM 32
// set HANG_TEST to 1 to demonstrate a hang for test purposes
#define HANG_TEST 0

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

static __device__ __inline__ uint32_t __smid(){
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;}

__device__ volatile int blocks_completed = 0;
// increase array length here if your GPU has more than 32 SMs
__device__ int first_SM[MAX_SM];

// launch with one thread per block only
__global__ void tkernel(int num_blocks, int num_SMs){

  int my_SM = __smid();
  int im_not_first = atomicCAS(first_SM+my_SM, 0, 1);
  if (!im_not_first){
    while (blocks_completed < (num_blocks-num_SMs+HANG_TEST));
  }
  atomicAdd((int *)&blocks_completed, 1);
}

int main(int argc, char *argv[]){
  unsigned my_dev = 0;
  if (argc > 1) my_dev = atoi(argv[1]);
  cudaSetDevice(my_dev);
  cudaCheckErrors("invalid CUDA device");

  int tot_SM = 0;
  cudaDeviceGetAttribute(&tot_SM, cudaDevAttrMultiProcessorCount, my_dev);
  cudaCheckErrors("CUDA error");
  if (tot_SM > MAX_SM) {
    printf("program configuration error\n");
    return 1;
  }
  printf("running on device %d, with %d SMs\n", my_dev, tot_SM);

  int temp[MAX_SM];
  for (int i = 0; i < MAX_SM; i++)
    temp[i] = 0;

  cudaMemcpyToSymbol(first_SM, temp, MAX_SM*sizeof(int));
  cudaCheckErrors("cudaMemcpyToSymbol fail");
  tkernel<<<NB, 1>>>(NB, tot_SM);
  cudaDeviceSynchronize();
  cudaCheckErrors("kernel error");
}
