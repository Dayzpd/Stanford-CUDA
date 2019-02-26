// Your parallel prefix sum implementation goes here.
// We've given you some skeleton code which you can fill in.
// The skeleton is just a suggestion -- programming ninjas
// feel free to do your own thing, but the interface in
// scan.h must be preserved.

#include "mp3-util.h"
#include <algorithm>
#include <iostream>
#include <math.h>

__global__
void scan(
  unsigned int *input, unsigned int *output, const size_t n, unsigned int *block_sums
) {
  extern __shared__ unsigned int pf_sum[];

  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int tid = threadIdx.x;

  pf_sum[tid] = input[gid];
  __syncthreads();

  for (int offset = 1; offset < n; offset *= 2)
  {
    if (tid >= offset)
    {
      pf_sum[tid] += pf_sum[tid - offset];
    }
    else
    {
      break;
    }
    __syncthreads();
  }

  output[gid] = pf_sum[tid];

  if ((gid + 1) % BLOCK_SIZE == 0)
  {
    block_sums[(gid + 1) / BLOCK_SIZE - 1] = pf_sum[tid];
  }
}

__global__
void scan_update(
  unsigned int *input, unsigned int *pf_sum, const size_t n,
  unsigned int *block_sums
) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < n)
  {
    for (size_t x = 0; x < blockIdx.x; x++)
    {
      //printf("id: %d, update: %d\n", id, block_sums[x]);
      pf_sum[id] = pf_sum[id] + block_sums[x];
    }

    pf_sum[id] -= input[id];
  }
}
