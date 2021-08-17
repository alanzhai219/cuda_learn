#include <cstdio>
#include <iostream>

int main(void)
{
    const int n = 32;
    const size_t sz = size_t(n) * sizeof(int);
    int *dJunk;
    cudaMalloc((void**)&dJunk, sz);
    cudaMemset(dJunk, 0, sz);
    cudaMemset(dJunk, 0x12, 32);

    int *Junk = new int[n];

    cudaMemcpy(Junk, dJunk, sz, cudaMemcpyDeviceToHost);

    for(int i=0; i<n; i++) {
        // fprintf(stdout, "%d %d\n", i, Junk[i]);
        std::cout << std::hex << Junk[i] << std::endl;
    }

    cudaDeviceReset();
    return 0;
}
