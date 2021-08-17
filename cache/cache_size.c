#include <iostream>
#include <random>
#include <ctime>
#include <algorithm>
#include <vector>


#define KB(x) ((size_t)(x) << 10)

int main() {
  std::vector<size_t> sizes_KB;
  for (int i=1; i<20; ++i) {
    sizes_KB.push_back(1<<i);
  }

  std::random_device rd;
  std::mt19937 gen(rd());

  for (size_t size : sizes_KB) {
    std::uniform_int_distribution<> dis(0, KB(size) -1);
    std::vector<char> memory(KB(size));
    std::fill(memory.begin(), memory.end(), 1);

    int dummy = 0;

    clock_t begin = clock();
    for (int i=0; i<(1<<25); ++i) {
      dummy += memory[dis(gen)];
    }
    clock_t end = clock();

    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << size << " KB, " << elapsed_secs << "secs, dummy: " << dummy << std::endl;
  }

  return 0;
}
