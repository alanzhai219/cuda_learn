#ifndef HEADER_GEMM_CUDA
#define HEADER_GEMM_CUDA

#include <memory>
#include <cuda_fp16.h>

// 行列積の計算
#ifdef USE_TENSORCORE
void productMatrixCudaWrap_TC(const std::size_t nA, const std::size_t mB, const std::size_t mA, const half* const dMatA, const half* const dMatB, half* const dMatC);
#else
void productMatrixCudaWrap_x2(const std::size_t nA, const std::size_t mB, const std::size_t mA, const half* const dMatA, const half* const dMatB, half* const dMatC);
#endif

#endif
