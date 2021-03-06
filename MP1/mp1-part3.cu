// This is machine problem 1, part 3, page ranking
// The problem is to compute the rank of a set of webpages
// given a link graph, aka a graph where each node is a webpage,
// and each edge is a link from one page to another.
// We're going to use the Pagerank algorithm (http://en.wikipedia.org/wiki/Pagerank),
// specifically the iterative algorithm for calculating the rank of a page
// We're going to run 20 iterations of the propage step.
// Implement the corresponding code in CUDA.

/* SUBMISSION GUIDELINES:
 * You should copy your entire device_graph_iterate fuction and the
 * supporting kernal into a file called mp1-part3-solution.cu and submit
 * that file. The fuction needs to have the exact same interface as the
 * device_graph_iterate function we provided. The kernel is internal
 * to your code and can look any way you want.
 */


#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <ctime>
#include <limits>

#include "mp1-util.h"

event_pair timer;

// amount of floating point numbers between answer and computed value
// for the answer to be taken correctly. 2's complement magick.
const int maxUlps = 1000;

void host_graph_propagate(
  unsigned int *graph_indices,
  unsigned int *graph_edges,
  float *graph_nodes_in,
  float *graph_nodes_out,
  float * inv_edges_per_node,
  int array_length
) {
  for(int i=0; i < array_length; i++)
  {
    float sum = 0.f;
    for(int j = graph_indices[i]; j < graph_indices[i+1]; j++)
    {
      sum += graph_nodes_in[graph_edges[j]]*inv_edges_per_node[graph_edges[j]];
    }
    graph_nodes_out[i] = 0.5f/(float)array_length + 0.5f*sum;
  }
}

void host_graph_iterate(
  unsigned int *graph_indices,
  unsigned int *graph_edges,
  float *graph_nodes_A,
  float *graph_nodes_B,
  float * inv_edges_per_node,
  int nr_iterations,
  int array_length
) {
  assert((nr_iterations % 2) == 0);
  for(int iter = 0; iter < nr_iterations; iter+=2)
  {
    host_graph_propagate(
      graph_indices,
      graph_edges,
      graph_nodes_A,
      graph_nodes_B,
      inv_edges_per_node,
      array_length
    );
    host_graph_propagate(
      graph_indices,
      graph_edges,
      graph_nodes_B,
      graph_nodes_A,
      inv_edges_per_node,
      array_length
    );
  }
}


// TODO your kernel code here
__global__ void device_graph_propagate(
  unsigned int *graph_indices,
  unsigned int *graph_edges,
  float *graph_nodes_in,
  float *graph_nodes_out,
  float * inv_edges_per_node,
  int array_length
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0;

  if (i < array_length)
  {
    for(int j = graph_indices[i]; j < graph_indices[i+1]; j++)
    {
      sum += graph_nodes_in[graph_edges[j]]*inv_edges_per_node[graph_edges[j]];
    }

    graph_nodes_out[i] = 0.5f/(float)array_length + 0.5f*sum;
  }
}

void device_graph_iterate(
  unsigned int *h_graph_indices,
  unsigned int *h_graph_edges,
  float *h_graph_nodes_input,
  float *h_graph_nodes_result,
  float *h_inv_edges_per_node,
  int nr_iterations,
  int num_elements,
  int avg_edges
) {
  unsigned int *d_graph_indices;
  unsigned int *d_graph_edges;
  float *d_graph_nodes_input;
  float *d_inv_edges_per_node;
  float *d_graph_nodes_result;

  int num_bytes_float = num_elements * sizeof(float);
  int num_bytes_ind = (num_elements + 1) * sizeof(unsigned int);
  int num_bytes_edg = num_elements * avg_edges * sizeof(unsigned int);

  // cudaMalloc device arrays
  cudaMalloc((void**)&d_graph_indices, num_bytes_ind);
  cudaMalloc((void**)&d_graph_edges, num_bytes_edg);
  cudaMalloc((void**)&d_graph_nodes_input, num_bytes_float);
  cudaMalloc((void**)&d_inv_edges_per_node, num_bytes_float);
  cudaMalloc((void**)&d_graph_nodes_result, num_bytes_float);

  // Report error if device memory allocation fails.
  if(
    d_graph_indices == 0 ||
    d_graph_edges == 0 ||
    d_graph_nodes_input == 0 ||
    d_graph_nodes_result == 0 ||
    d_inv_edges_per_node == 0
  ) {
    printf("couldn't allocate memory\n");
    exit(1);
  }

  int block_size = 512;
  int grid_size = num_elements / block_size;

  start_timer(&timer);

  // Copy input data to device
  cudaMemcpy(d_graph_indices, h_graph_indices, num_bytes_ind, cudaMemcpyHostToDevice);
  cudaMemcpy(d_graph_edges, h_graph_edges, num_bytes_edg, cudaMemcpyHostToDevice);
  cudaMemcpy(d_graph_nodes_input, h_graph_nodes_input, num_bytes_float, cudaMemcpyHostToDevice);
  cudaMemcpy(d_inv_edges_per_node, h_inv_edges_per_node, num_bytes_float, cudaMemcpyHostToDevice);

  // TODO your device code should go here, so you can measure how long it takes
  for(int iter = 0; iter < nr_iterations; iter+=2)
  {
    device_graph_propagate<<<grid_size, block_size>>>(
      d_graph_indices,
      d_graph_edges,
      d_graph_nodes_input,
      d_graph_nodes_result,
      d_inv_edges_per_node,
      num_elements
    );

    device_graph_propagate<<<grid_size, block_size>>>(
      d_graph_indices,
      d_graph_edges,
      d_graph_nodes_result,
      d_graph_nodes_input,
      d_inv_edges_per_node,
      num_elements
    );
  }

  check_launch("gpu graph propagate");
  stop_timer(&timer,"gpu graph propagate");

  // TODO your final result should end up in h_graph_nodes_result, which is a *host* pointer
  cudaMemcpy(h_graph_nodes_result, d_graph_nodes_result, num_bytes_float, cudaMemcpyDeviceToHost);

  cudaFree(d_graph_indices);
  cudaFree(d_inv_edges_per_node);
  cudaFree(d_graph_edges);
  cudaFree(d_graph_nodes_input);
  cudaFree(d_graph_nodes_result);
}


int main(void)
{
  // create arrays of 2M elements
  int num_elements = 1 << 21;
  int avg_edges = 8;
  int iterations = 20;

  // pointers to host & device arrays
  unsigned int *h_graph_indices = 0;
  float *h_inv_edges_per_node = 0;
  unsigned int *h_graph_edges = 0;
  float *h_graph_nodes_input = 0;
  float *h_graph_nodes_result = 0;
  float *h_graph_nodes_checker_A = 0;
  float *h_graph_nodes_checker_B = 0;


  // malloc host array
  // index array has to be n+1 so that the last thread can
  // still look at its neighbor for a stopping point
  h_graph_indices = (unsigned int*)malloc((num_elements+1) * sizeof(unsigned int));
  h_inv_edges_per_node = (float*)malloc((num_elements) * sizeof(float));
  h_graph_edges = (unsigned int*)malloc(num_elements * avg_edges * sizeof(unsigned int));
  h_graph_nodes_input = (float*)malloc(num_elements * sizeof(float));
  h_graph_nodes_result = (float*)malloc(num_elements * sizeof(float));
  h_graph_nodes_checker_A = (float*)malloc(num_elements * sizeof(float));
  h_graph_nodes_checker_B = (float*)malloc(num_elements * sizeof(float));

  // if any memory allocation failed, report an error message
  if(h_graph_indices == 0 || h_graph_edges == 0 || h_graph_nodes_input == 0 || h_graph_nodes_result == 0 ||
	 h_inv_edges_per_node == 0 || h_graph_nodes_checker_A == 0 || h_graph_nodes_checker_B == 0)
  {
    printf("couldn't allocate memory\n");
    exit(1);
  }

  // generate random input
  // initialize
  srand(time(NULL));

  h_graph_indices[0] = 0;
  for(int i=0;i< num_elements;i++)
  {
    int nr_edges = (i % 15) + 1;
    h_inv_edges_per_node[i] = 1.f/(float)nr_edges;
    h_graph_indices[i+1] = h_graph_indices[i] + nr_edges;
    if(h_graph_indices[i+1] >= (num_elements * avg_edges))
    {
      printf("more edges than we have space for\n");
      exit(1);
    }
    for(int j=h_graph_indices[i];j<h_graph_indices[i+1];j++)
    {
      h_graph_edges[j] = rand() % num_elements;
    }

    h_graph_nodes_input[i] =  1.f/(float)num_elements;
    h_graph_nodes_checker_A[i] =  h_graph_nodes_input[i];
    h_graph_nodes_result[i] = std::numeric_limits<float>::infinity();
  }

  device_graph_iterate(
    h_graph_indices,
    h_graph_edges,
    h_graph_nodes_input,
    h_graph_nodes_result,
    h_inv_edges_per_node,
    iterations,
    num_elements,
    avg_edges
  );

  start_timer(&timer);
  // generate reference output
  host_graph_iterate(
    h_graph_indices,
    h_graph_edges,
    h_graph_nodes_checker_A,
    h_graph_nodes_checker_B,
    h_inv_edges_per_node,
    iterations,
    num_elements
  );

  check_launch("host graph propagate");
  stop_timer(&timer,"host graph propagate");

  // check CUDA output versus reference output
  int error = 0;
  int num_errors = 0;
  for(int i=0;i<num_elements;i++)
  {
    float n = h_graph_nodes_result[i];
    float c = h_graph_nodes_checker_A[i];
    if(!AlmostEqual2sComplement(n,c,maxUlps))
    {
      num_errors++;
      if (num_errors < 10)
      {
            printf("%d:%.3f::",i, n-c);
      }
      error = 1;
    }
  }

  if(error)
  {
    printf("Output of CUDA version and normal version didn't match! \n");
  }
  else
  {
    printf("Worked! CUDA and reference output match. \n");
  }

  // deallocate memory
  free(h_graph_indices);
  free(h_inv_edges_per_node);
  free(h_graph_edges);
  free(h_graph_nodes_input);
  free(h_graph_nodes_result);
  free(h_graph_nodes_checker_A);
  free(h_graph_nodes_checker_B);
}
