#include <iostream>
#include <stdio.h>

#include <cuda_runtime.h>

__global__ void warp_ballot() {
    int tid = threadIdx.x;
    unsigned mask = 0xffffffff;
    unsigned ret = __ballot_sync(mask, tid>44 && tid<46);
    printf("tid = %d, ret = %x\n", tid, ret);
}
int main(int argc, char* argv[]) {
    warp_ballot<<<1,48>>>();
    cudaDeviceSynchronize();
    return 0;
}