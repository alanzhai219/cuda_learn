#include <stdio.h>

__global__ void bcast(int* data) {
    int tid = threadIdx.x & 0x1f;
    int value = data[tid];
    unsigned FULL_MASK = 0XFFFFFFFF;
    unsigned mask = __ballot_sync(__activemask(), tid %2);
    printf("mask----%x\n", mask);
    printf("activemask----%x\n", __activemask());
    if (tid % 2) {
      value = __shfl_sync(mask, value, 5); 
    } else {
      value = __shfl_sync(FULL_MASK - mask, value, 0);
    }
    data[tid] = value;
}

int main() {
    int* h_data = NULL;
    int* d_data = NULL;

    h_data = (int*)malloc(32 * sizeof(int));
    for(int i=0; i<32; i++) {
      h_data[i] = i;
      printf("%2d ", h_data[i]);
    }
    printf("\n");

    cudaMalloc((void**)&d_data, 32 * sizeof(int));
    cudaMemcpy(d_data, h_data, 32 * sizeof(int), cudaMemcpyHostToDevice);

    bcast<<< 1, 32 >>>(d_data);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0; i<32; i++) {
      printf("%2d ", h_data[i]);
    }
    printf("\n");

    return 0;
}
