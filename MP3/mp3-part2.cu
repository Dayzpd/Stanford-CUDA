#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <limits>
#include <algorithm>
#include "math.h"

#define BLOCK_SIZE 64
#include "scatter.cu"
#include "scan.cu"
#include "compact.cu"


#include "mp3-util.h"


int main(void)
{
  size_t n = 1024;

  unsigned int *h_num_compacted = 0;
  unsigned int *d_num_compacted = 0;
  h_num_compacted = (unsigned int*)malloc(sizeof(unsigned int));
  cudaMalloc((void**)&d_num_compacted, sizeof(unsigned int));
  cudaMemset(d_num_compacted, 0, sizeof(unsigned int));

  unsigned int *h_input = 0;
  unsigned int *d_input = 0;
  h_input = (unsigned int*)malloc(n * sizeof(unsigned int));
  cudaMalloc((void**)&d_input, n * sizeof(unsigned int));

  for(int i=0;i < n;i++)
  {
    h_input[i] = rand();
  }

  cudaMemcpy(
    d_input, h_input, sizeof(unsigned int) * n, cudaMemcpyHostToDevice
  );

  unsigned int *h_bit_vector = 0;
  unsigned int *d_bit_vector = 0;
  h_bit_vector = (unsigned int*)malloc(n * sizeof(unsigned int));
  cudaMalloc((void**)&d_bit_vector, n * sizeof(unsigned int));
  cudaMemset(d_bit_vector, 0, n * sizeof(unsigned int));

  unsigned int *h_block_sums = 0;
  unsigned int *d_block_sums = 0;
  h_block_sums = (unsigned int*)malloc(ceil(n / BLOCK_SIZE) * sizeof(unsigned int));
  cudaMalloc((void**)&d_block_sums, ceil(n / BLOCK_SIZE) * sizeof(unsigned int));

  unsigned int *h_prefix_sum = 0;
  unsigned int *d_prefix_sum = 0;
  h_prefix_sum = (unsigned int*)malloc(n * sizeof(unsigned int));
  cudaMalloc((void**)&d_prefix_sum, n * sizeof(unsigned int));

  test_bit_vector<<<n / BLOCK_SIZE, BLOCK_SIZE>>>(
    d_bit_vector, d_input, d_num_compacted, n
  );

  cudaMemcpy(
    h_bit_vector, d_bit_vector, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost
  );

  scan<<<n / BLOCK_SIZE, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(unsigned int)>>>(
    d_bit_vector, d_prefix_sum, n, d_block_sums
  );
  check_cuda_error("scan", __FILE__, __LINE__);

  scan_update<<<n / BLOCK_SIZE, BLOCK_SIZE>>>(d_bit_vector, d_prefix_sum, n, d_block_sums);
  check_cuda_error("update", __FILE__, __LINE__);

  cudaMemcpy(
    h_prefix_sum, d_prefix_sum, sizeof(unsigned int) * n,
    cudaMemcpyDeviceToHost
  );

  cudaMemcpy(
    h_block_sums, d_block_sums, ceil(n / BLOCK_SIZE) * sizeof(unsigned int),
    cudaMemcpyDeviceToHost
  );

  for (size_t i = 0; i < n; i++)
  {
    printf("%d, %d\n", h_bit_vector[i], h_prefix_sum[i]);
  }

  cudaFree(d_num_compacted);
  cudaFree(d_input);
  cudaFree(d_bit_vector);
  cudaFree(d_block_sums);
  cudaFree(d_prefix_sum);

  free(h_num_compacted);
  free(h_input);
  free(h_bit_vector);
  free(h_block_sums);
  free(h_prefix_sum);

  return 0;
}
