#include <iostream>

void scan_serialize(int *output, int *input, int len) {
  int temp = input[0];
  for (int i=1; i<len; ++i) {
    temp += input[i];
    output[i] = temp; 
  }
}

int main (int argc, char* argv[]) {
  const int num = 8;
  int arr_in[num] = {3, 1, 7, 0, 4, 1, 6, 3};
  int arr_out[num] = {0};
  // call scan func
  scan_serialize(arr_out, arr_in, num);
  // dump out
  for (auto a : arr_out)
    std::cout << a << std::endl;

  return 0;
}
