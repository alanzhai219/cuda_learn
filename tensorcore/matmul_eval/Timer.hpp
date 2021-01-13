#ifndef TIMER_HPP_INCLUDED
#define TIMER_HPP_INCLUDED

#include <chrono>

// 時間計測用
class Timer
{
	std::chrono::time_point<std::chrono::system_clock> begin;

public:
	void start()
	{
		this->begin = std::chrono::system_clock::now();
	}

	template<typename T = std::chrono::microseconds>
	auto time()
	{
		const auto end = std::chrono::system_clock::now();
		return std::chrono::duration_cast<T>(end - begin).count();
	}
};
#endif
