#ifndef HEADER_PROBLEM
#define HEADER_PROBLEM

#include <iostream>
#include <random>
#include <memory>
#include <cuda_fp16.h>

#include "GemmCpuNaive.hpp"

#include "Timer.hpp"

namespace Gemm
{

	template<typename T>
	using ConstPointer = std::add_pointer_t<std::add_const_t<std::remove_reference_t<T>>>;
	
	class Problem
	{
	public:

		const std::size_t nA;
		const std::size_t mB;
		const std::size_t mA;
		
		const std::unique_ptr<half[]> matA = std::make_unique<half[]>(nA * mA); // 本当はconst std::unique_ptr<const float[]>にしたい
		const std::unique_ptr<half[]> matB = std::make_unique<half[]>(mA * mB);

		const std::unique_ptr<half[]> matC = std::make_unique<half[]>(nA * mB); // 解行列

		const std::size_t opNum; // 演算の合計数（積演算の数＋和演算の数）
		
		// nA*mA行列matAとmA*mB行列matBを初期化する
		Problem(const std::size_t nA, const std::size_t mB, const std::size_t mA)
			: nA(nA), mB(mB), mA(mA), opNum(nA * mA * mB + nA * (mA - 1) * mB)
		{
			// 行列の初期化
			initMatrix(matA.get(), nA, mA);
			initMatrix(matB.get(), mA, mB);

			// 解行列の生成
			makeAnswer();
		}
		
		// half型のnA*mA行列のmatAを乱数で初期化する
		static void initMatrix(half* const matA, const std::size_t nA, const std::size_t mA)
		{
			// 32bit版メルセンヌ・ツイスタの乱数生成
			std::random_device rnd; // シード生成用の非決定的な乱数生成期
			std::mt19937 mt(rnd()); // メルセンヌ・ツイスタ32bit
			std::uniform_real_distribution<float> mt32(0.0f, 1.0f); // 生成範囲の指定
			
			// 乱数で行列の全要素を決定
			for (auto i = decltype(nA)(0); i < nA; ++i)
			{
				for (auto j = decltype(mA)(0); j < mA; ++j)
				{
					matA[i * mA + j] = __float2half(mt32(mt));
				}
			}
		}

		auto checkAnswer(const half* const matC) const
		{
			// 誤差を計算
			// 全要素の相対誤差の最大値
			float err = 0.0f;
			for (auto i = decltype(nA)(0); i < nA; ++i)
			{
				for (auto j = decltype(mB)(0); j < mB; ++j)
				{
					const auto ind = i * mB + j;
					const float expected = __half2float(this->matC[ind]);
					const float computed = __half2float(matC[ind]);
					err = std::max(err,std::abs(expected - computed)/expected);
				}
			}

			return err;
		}

		template<typename PRE, typename GEMM, typename POST>
		auto benchmark(PRE&& pre, GEMM&& gemm, POST&& post) const
		{
			// 解保存用メモリ
			auto matRes = std::make_unique<half[]>(nA * mB);
			
			// 前処理
			pre(nA, mB, mA, matA.get(), matB.get());
			
			// 時間計測開始
			Timer timer;
			timer.start();

			// 行列の積の計算
			gemm(nA, mB, mA, matA.get(), matB.get(), matRes.get());

			// 計測時間からGFLOPSを計算
			const auto usec = timer.time<std::chrono::microseconds>();
			const auto gflops = usec != decltype(usec)(0) ? static_cast<double>(opNum) / static_cast<double>(usec) * 1.0e-3 : 0.0;
			
			// 後処理
			post(nA, mB, matRes.get());

			// 誤差の計算
			const auto err = checkAnswer(matRes.get());
			
			// ベンチマーク結果表示
			// 計算時間を表示
			std::cout << usec * 1e-3 << "[ms]" << std::endl;
			// GFLOPSを表示
			std::cout << gflops << "[GFLOPS]" << std::endl;
			// 誤差を表示
			std::cout << "Evaluated error: " << err * 100 << "%" << std::endl;
			
			return std::move(matRes);
		}

		void makeAnswer()
		{
			// 正解の作成
			productMatrix(nA, mB, mA, matA.get(), matB.get(), matC.get());
		}
	};
}

#endif
