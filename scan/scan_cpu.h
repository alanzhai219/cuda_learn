void scan_cpu(int *output, int *input, int len) {
  int temp = input[0];
  for (int i=1; i<len; ++i) {
    temp += input[i];
    output[i] = temp; 
  }
}
