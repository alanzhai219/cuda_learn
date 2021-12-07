#include <mma.h>
#include <iostream>

using namespace nvcuda;

__global__ void wmma_ker(half *a, half *b, float *c) {
   // Declare the fragments
   /*
    template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;
    Layout: use to transpose
   */
   wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag; // row major
   wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag; // col major
   wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

   // Initialize the output to zero
   wmma::fill_fragment(c_frag, 0.0f);

   // Load the inputs
   int warpX = threadIdx.x / 32;
   // 内存在物理上都是线性排列，row major或col major是逻辑概念
   // 如果A是row major，且a_frag也是row major，则load_matrix_sync不需要做transpose
   /*
      -------------------------------------
      |  frag \ A | row_major | col_major |
      -------------------------------------
      | row_major |     N     |      Y    |
      -------------------------------------  
      | col_major |     Y     |      N    |
      -------------------------------------
   */
   // printf("warpX = %d\n", warpX);
   wmma::load_matrix_sync(a_frag, a, 16);
   wmma::load_matrix_sync(b_frag, b+16*warpX, 16);

   // Perform the matrix multiplication
   // 按照传统“矩阵乘法”，所以a_frag和b_frag的数据分布，直接影响c_frag的结果
   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag); 

   // Store the output
   /*
    void store_matrix_sync(T* mptr, const fragment<...> &a, unsigned ldm, layout_t layout);
    - mptr: 需要存储的内存的起始地址
    - a: 待存储的矩阵块
    - ldm: mptr所对应地址的ldm
    - layout: mptr存储的形状
   */
   wmma::store_matrix_sync(c+16*warpX, c_frag, 16, wmma::mem_row_major);
}

template <typename T>
void dump_matrix(T *data_ptr, size_t M, size_t N) {
  for (int i=0; i<M; ++i) {
    for (int j=0; j<N; ++j) {
      std::cout << static_cast<float>(data_ptr[i*N+j]) << ", ";
    }
    std::cout << "\n";
  }
}
int main(){

  const int M = 16;
  const int N = 16;
  const int K = 16;

  half *d_a, *h_a, *d_b, *h_b;
  float *d_c, *h_c;
  // type of a and b is half
  h_a = new half[M*K];
  h_b = new half[K*N];

  // type of c is float.
  h_c = new float[M*N];

  cudaMalloc(&d_a, M*K*sizeof(half));
  cudaMalloc(&d_b, K*N*sizeof(half));
  cudaMalloc(&d_c, M*N*sizeof(float));
  for (int i = 0; i < M*K; i++) {
    h_a[i] = 0.0f + i;
  }
  for (int i = 0; i < K*N; i++) {
    h_b[i] = 0.0f + i;
  }
  dump_matrix(h_a, M, K);
  dump_matrix(h_b, K, N);
  cudaMemcpy(d_a, h_a, M*K*sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, K*N*sizeof(half), cudaMemcpyHostToDevice);
  wmma_ker<<<1,32>>>(d_a, d_b, d_c);
  cudaMemcpy(h_c, d_c, M*N*sizeof(float), cudaMemcpyDeviceToHost);
  dump_matrix(h_c, M, N);
  return 0;
}
