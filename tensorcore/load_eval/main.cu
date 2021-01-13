#include <memory>
#include <cuda_fp16.h>
#include <mma.h>

#define USE_WMMA

// 行列の1辺の大きさ
constexpr std::size_t MATRIX_DIM = 16;

// 読み込み関数呼び出し回数
constexpr std::size_t NUM_LOADFUNC_CALL = 1 << 20;

namespace
{
template <class T>
struct DeviceDeleter
{
	void operator()(T* ptr)
	{
		cudaFree(ptr);
	}
};

template <class T>
auto getDevicePtr(const std::size_t size)
{
	T* ptr;
	cudaMalloc((void**)&ptr, sizeof(T) * size);
	return std::unique_ptr<T, DeviceDeleter<T>>{ptr};
}

// 自作のfragment読み込み関数
__device__ void load_16x16_matrix(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, MATRIX_DIM, MATRIX_DIM, MATRIX_DIM, half, nvcuda::wmma::col_major> &fragment, const half* const memory_ptr)
{
	const auto warp_id = (blockIdx.x * blockDim.x + threadIdx.x) % warpSize;
	const auto warp_mat_cols = (warp_id % 8) / 4 * 2 + (warp_id / 16);
#pragma unroll
	for(int i = 0; i < 4; i++)
	{
		const auto load_mat_rows = 4 * i + warp_id % 4;
#pragma unroll
		for(int j = 0; j < 4; j++)
		{
			const auto load_mat_cols = 4 * warp_mat_cols + j;
			fragment.x[i * 4 + j] = __ldg(memory_ptr + MATRIX_DIM * load_mat_rows + load_mat_cols);
		}
		__syncthreads();
	}
}

// fragment読み込み関数の速度比較用のカーネル関数
__global__ void load_matrix_kernel(const half* const dA, const half* const dB)
{
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, MATRIX_DIM, MATRIX_DIM, MATRIX_DIM, half, nvcuda::wmma::col_major> frag_c;
	for(auto i = decltype(NUM_LOADFUNC_CALL)(0); i < NUM_LOADFUNC_CALL; i++)
#ifdef USE_WMMA
		nvcuda::wmma::load_matrix_sync(frag_c, dA, MATRIX_DIM);
#else
		load_16x16_matrix(frag_c, dA);
#endif
	__syncthreads();
}
}



int main()
{
	auto dMatA = getDevicePtr<half>(MATRIX_DIM * MATRIX_DIM);
	auto dMatB = getDevicePtr<half>(MATRIX_DIM * MATRIX_DIM);

	constexpr std::size_t warpSize = 32;
	load_matrix_kernel<<<1, warpSize>>>(dMatA.get(), dMatB.get());
}
