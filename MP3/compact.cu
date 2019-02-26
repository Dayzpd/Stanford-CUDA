#include "compact.h"
#include "scatter.h"
#include "mp3-util.h"
#include "math.h"


__global__
void create_bit_vector(
  unsigned int* d_bit_vector, const real* d_call, const real* d_put,
  size_t* num_compacted, const size_t n, const real min_call_threshold,
  const real min_put_threshold
) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
  {
    if (d_call[i] >= min_call_threshold && d_put[i] >= min_put_threshold)
    {
      d_bit_vector[i] = 1;
      atomicAdd(num_compacted, 1);

    }
    //printf("%d, %d\n", d_call[i], d_put[i]);
  }
}

__global__
void test_bit_vector(
  unsigned int* d_bit_vector, const unsigned int *d_input,
  unsigned int *num_compacted, const size_t n
) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n)
  {
    if (d_input[i] % 2 == 0)
    {
      d_bit_vector[i] = 1;
      atomicAdd(num_compacted, 1);
    }
  }
}


// compact_options copies the input options whose call and put
// results from the first round meet or exceed the given call & put
// thresholds to a compacted output in three result arrays.
size_t compact_options(
  const real min_call_threshold, const real min_put_threshold,
  const real *d_call, const real *d_put, const real *d_stock_price_input,
  const real *d_option_strike_input, const real *d_option_years_input,
  const size_t n, real *d_stock_price_result, real *d_option_strike_result,
  real *d_option_years_result
) {

  // Keeps track of how many options are compacted
  size_t h_num_compacted = 0;
  size_t *d_num_compacted = 0;
  cudaMalloc((void**)&d_num_compacted, sizeof(size_t));
  cudaMemset(d_num_compacted, 0, sizeof(size_t));
  check_cuda_error("Malloc (d_num_compacted)", __FILE__, __LINE__);

  // Holds bit vector of options that meet call/put minimum thresholds
  unsigned int *h_bit_vector = 0;
  unsigned int *d_bit_vector = 0;
  h_bit_vector = (unsigned int*)malloc(n * sizeof(unsigned int));
  cudaMalloc((void**)&d_bit_vector, n * sizeof(unsigned int));
  cudaMemset(d_bit_vector, 0, n * sizeof(unsigned int));
  check_cuda_error("Malloc (d_bit_vector)", __FILE__, __LINE__);

  // Holds the sums of each section of the bit vector
  unsigned int *d_block_sums = 0;
  cudaMalloc((void**)&d_block_sums, (ceil((n) / BLOCK_SIZE)) * sizeof(unsigned int));
  check_cuda_error("Malloc (d_block_sums)", __FILE__, __LINE__);

  // Holds the final prefix sum calculated using the Hillis and Steele algorithm
  unsigned int *d_prefix_sum = 0;
  cudaMalloc((void**)&d_prefix_sum, n * sizeof(unsigned int));
  check_cuda_error("Malloc (d_prefix_sum)", __FILE__, __LINE__);

  // Calculate number of threadblocks
  size_t num_blocks = (n / BLOCK_SIZE);

  create_bit_vector<<<num_blocks, BLOCK_SIZE>>>(
    d_bit_vector, d_call, d_put, d_num_compacted, n, min_call_threshold, min_put_threshold
  );
  check_cuda_error("bit_vector", __FILE__, __LINE__);

  scan<<<num_blocks, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(unsigned int)>>>(
    d_bit_vector, d_prefix_sum, n, d_block_sums
  );
  check_cuda_error("scan", __FILE__, __LINE__);

  scan_update<<<num_blocks, BLOCK_SIZE>>>(d_bit_vector, d_prefix_sum, n, d_block_sums);
  check_cuda_error("scan_update", __FILE__, __LINE__);

  scatter_options_kernel<<<num_blocks, BLOCK_SIZE>>>(
    min_call_threshold, min_put_threshold, d_call,
    d_put, d_stock_price_input, d_option_strike_input,
    d_option_years_input, d_prefix_sum, n, d_stock_price_result,
    d_option_strike_result, d_option_years_result
  );
  check_cuda_error("scan_update", __FILE__, __LINE__);

  cudaMemcpy(
    &h_num_compacted, d_num_compacted, sizeof(unsigned int), cudaMemcpyDeviceToHost
  );

  cudaFree(d_num_compacted);
  cudaFree(d_bit_vector);
  cudaFree(d_block_sums);
  cudaFree(d_prefix_sum);
  free(h_bit_vector);

  return h_num_compacted;
}
