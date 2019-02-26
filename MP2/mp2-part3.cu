#include <cassert>

#define BLOCK_SIZE 256
#define NUM_NEIGHBORS 4

#include "mp2.h"
#include "mp2-part3-util.h"
#include "mp2-util.h"

event_pair timer;

//------------------------------------------------------------------------------

void host_knn_particle(
  float3 *particles, int *bins, int *part_knn, int id, int bin_id, int bx,
  int by, int bz, int3 binning_dim, int bin_size
) {
  // for each particle
  // loop over all the neighbor bins in x,y and z,
  // as well as the bin it is in itself

  float neigh_dist[NUM_NEIGHBORS];
  int neigh_ids[NUM_NEIGHBORS];

  init_list(&neigh_dist[0],NUM_NEIGHBORS,2.0f);
  init_list(&neigh_ids[0],NUM_NEIGHBORS,-1);

  float3 pos = particles[id];

  for(int x_offset=-1;x_offset<2;x_offset++)
  {
    int nx = bx + x_offset;
    if(nx > -1 && nx < binning_dim.x)
    {
      for(int y_offset=-1;y_offset<2;y_offset++)
      {
        int ny = by + y_offset;
        if(ny > -1 && ny < binning_dim.y)
        {
          for(int z_offset=-1;z_offset<2;z_offset++)
          {
            int nz = bz + z_offset;
            if(nz > -1 && nz < binning_dim.z)
            {
              int neigh_bin_id = nx + binning_dim.x*(ny + binning_dim.y*nz);

              // loop over all the particles in those bins
              for(int bin_offset=0;bin_offset<bin_size;bin_offset++)
              {
                int neigh_particle_id = bins[neigh_bin_id*bin_size + bin_offset];
                // skip empty bin entries and don't interact with yourself

                if(neigh_particle_id != -1 && neigh_particle_id != id)
                {
                  float rsq = dist2(pos,particles[neigh_particle_id]);
                  insert_list(
                    &neigh_dist[0], &neigh_ids[0], rsq,
                    neigh_particle_id
                  );
                }
              }
            }
          }
        }
      }
    }
  }
  for(int j=0;j<NUM_NEIGHBORS;j++)
  {
    part_knn[j] = neigh_ids[j];
  }
}

//------------------------------------------------------------------------------

__device__
void device_knn_particle(
  float3 *particles, int *bins, int *part_knn, int id, int bin_id, int bx,
  int by, int bz, int3 binning_dim, int bin_size
) {
  // for each particle
  // loop over all the neighbor bins in x,y and z,
  // as well as the bin it is in itself

  float neigh_dist[NUM_NEIGHBORS];
  int neigh_ids[NUM_NEIGHBORS];

  init_list(&neigh_dist[0],NUM_NEIGHBORS,2.0f);
  init_list(&neigh_ids[0],NUM_NEIGHBORS,-1);

  float3 pos = particles[id];

  for(int x_offset=-1;x_offset<2;x_offset++)
  {
    int nx = bx + x_offset;
    if(nx > -1 && nx < binning_dim.x)
    {
      for(int y_offset=-1;y_offset<2;y_offset++)
      {
        int ny = by + y_offset;
        if(ny > -1 && ny < binning_dim.y)
        {
          for(int z_offset=-1;z_offset<2;z_offset++)
          {
            int nz = bz + z_offset;
            if(nz > -1 && nz < binning_dim.z)
            {
              int neigh_bin_id = nx + binning_dim.x*(ny + binning_dim.y*nz);

              // loop over all the particles in those bins
              for(int bin_offset=0;bin_offset<bin_size;bin_offset++)
              {
                int neigh_particle_id = bins[neigh_bin_id*bin_size + bin_offset];
                // skip empty bin entries and don't interact with yourself
                __syncthreads();
                if(neigh_particle_id != -1 && neigh_particle_id != id)
                {
                  __syncthreads();
                  float rsq = dist2(pos,particles[neigh_particle_id]);
                  insert_list(
                    &neigh_dist[0], &neigh_ids[0], rsq,
                    neigh_particle_id
                  );
                  __syncthreads();
                }
                __syncthreads();
              }
            }
          }
        }
      }
    }
  }
  __syncthreads();
  for(int j=0;j<NUM_NEIGHBORS;j++)
  {
    __syncthreads();
    part_knn[j] = neigh_ids[j];
  }
  __syncthreads();
}

//------------------------------------------------------------------------------

void host_binned_knn(
  float3 *particles, int *bins, int *knn, int3 binning_dim, int bin_size
) {
  // loop over all bins
  for(int bx=0;bx<binning_dim.x;bx++)
  {
    for(int by=0;by<binning_dim.y;by++)
    {
      for(int bz=0;bz<binning_dim.z;bz++)
      {
        int bin_id = bx + binning_dim.x*(by + binning_dim.y*bz);

        for(int j=0;j<bin_size;j++)
        {
          int id = bins[bin_id*bin_size + j];
          if(id != -1)
          {
            host_knn_particle(
              particles, bins, &knn[id*NUM_NEIGHBORS],id, bin_id, bx, by, bz,
              binning_dim, bin_size
            );
          }
        }
      }
    }
  }
}

//------------------------------------------------------------------------------

__global__
void device_binned_knn(
  float3 *particles, int *bins, int *knn, int3 binning_dim, int bin_size
) {
  int bx = blockIdx.x * blockDim.x + threadIdx.x;
  int by = blockIdx.y * blockDim.y + threadIdx.y;
  int bz = blockIdx.z * blockDim.z + threadIdx.z;

  if (bx < binning_dim.x && by < binning_dim.y && bz < binning_dim.z){
    int bin_id = bx + binning_dim.x*(by + binning_dim.y*bz);

    for(int j=0; j<bin_size; j++)
    {
      int id = bins[bin_id * bin_size + j];

      if(id != -1)
      {
        device_knn_particle(
          particles, bins, &knn[id * NUM_NEIGHBORS], id, bin_id, bx, by, bz,
          binning_dim, bin_size
        );
      }
    }
  }
}

//------------------------------------------------------------------------------

void host_binning(
  float3 *particles, int *bins, int *bin_counters, int3 gridding, int bin_size,
  int num_particles
) {

  for (int i=0; i<num_particles; i++)
  {
    unsigned int bin = bin_index(particles[i],gridding);
    if(bin_counters[bin] < bin_size)
    {
      unsigned int offset = bin_counters[bin];
      bin_counters[bin]++;
      bins[bin * bin_size + offset] = i;
    }
  }
}

//------------------------------------------------------------------------------

__global__
void device_binning(
  float3 * d_particles, int * d_bins, int * d_bin_counters, int3 gridding,
  int bin_size, int num_particles
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < num_particles)
  {
    unsigned int bin = bin_index(d_particles[i], gridding);
    if(d_bin_counters[bin] < bin_size)
    {
      int offset = atomicAdd(&d_bin_counters[bin], 1);
      d_bins[bin * bin_size + offset] = i;
    }
  }
}

//------------------------------------------------------------------------------

void allocate_host_memory(
  int num_particles, int num_bins, int bin_size, float3 *&h_particles,
  float3 *&h_particles_checker, int *&h_bins, int *&h_bins_checker,
  int *&h_bin_counters, int *&h_bin_counters_checker, int *&h_knn,
  int *&h_knn_checker
) {
  h_particles = (float3*)malloc(num_particles * sizeof(float3));
  h_particles_checker = (float3*)malloc(num_particles * sizeof(float3));

  h_bins = (int*)malloc(num_bins * bin_size * sizeof(int));
  h_bins_checker = (int*)malloc(num_bins * bin_size * sizeof(int));

  h_bin_counters = (int*)malloc(num_bins * sizeof(int));
  h_bin_counters_checker = (int*)malloc(num_bins * sizeof(int));

  h_knn = (int*)malloc(num_particles * NUM_NEIGHBORS * sizeof(int));
  h_knn_checker = (int*)malloc(num_particles * NUM_NEIGHBORS * sizeof(int));
}

//------------------------------------------------------------------------------

void allocate_device_memory(
  int num_particles, int num_bins, int bin_size, float3 *&d_particles,
  int *&d_bins, int *&d_knn, int *&d_bin_counters
) {
  cudaMalloc((void**)&d_particles, num_particles * sizeof(float3));
  cudaMalloc((void**)&d_bins, num_bins * bin_size * sizeof(int));
  cudaMalloc((void**)&d_knn, NUM_NEIGHBORS * num_particles * sizeof(int));
  cudaMalloc((void**)&d_bin_counters, num_bins * sizeof(int));
}

//------------------------------------------------------------------------------

void deallocate_host_memory(
  float3 *h_particles, int *h_bins, int *h_knn, int *h_bin_counters,
  float3 *h_particles_checker, int *h_bins_checker, int *h_knn_checker,
  int *h_bin_counters_checker
) {
  // deallocate memory
  free(h_particles);
  free(h_bins);
  free(h_knn);
  free(h_bin_counters);
  free(h_particles_checker);
  free(h_bins_checker);
  free(h_knn_checker);
  free(h_bin_counters_checker);
}

//------------------------------------------------------------------------------

void deallocate_device_memory(
  float3 *d_particles, int *d_bins, int *d_knn, int *d_bin_counters
) {
  cudaFree(d_particles);
  cudaFree(d_bins);
  cudaFree(d_knn);
  cudaFree(d_bin_counters);
}

//------------------------------------------------------------------------------

int main(void)
{
  // Hyperparameters
  int num_particles = 64*1024;
  int log_bpd = 4;
  int bins_per_dim = 1 << log_bpd;
  unsigned int num_bins = bins_per_dim * bins_per_dim * bins_per_dim;
  int bin_size = num_particles/num_bins * 3;
  int3 gridding = make_int3(log_bpd, log_bpd, log_bpd);
  int3 binning_dim = make_int3(bins_per_dim,bins_per_dim,bins_per_dim);

  float3 *h_particles = 0;
  int *h_bins = 0;
  int *h_bin_counters = 0;
  int *h_bins_checker = 0;
  float3 *h_particles_checker = 0;
  int *h_bin_counters_checker = 0;
  int *h_knn = 0;
  int *h_knn_checker = 0;

  float3 *d_particles = 0;
  int *d_bins = 0;
  int *d_knn = 0;
  int *d_bin_counters = 0;

  allocate_host_memory(
    num_particles, num_bins, bin_size, h_particles,
    h_particles_checker, h_bins, h_bins_checker,
    h_bin_counters, h_bin_counters_checker, h_knn,
    h_knn_checker
  );

  allocate_device_memory(
    num_particles, num_bins, bin_size, d_particles, d_bins, d_knn,
    d_bin_counters
  );

  // generate random input
  // initialize
  srand(13);

  for(int i=0;i< num_particles;i++)
  {
    h_particles[i] = h_particles_checker[i] = make_float3(
      (float)rand()/(float)RAND_MAX,
      (float)rand()/(float)RAND_MAX,
      (float)rand()/(float)RAND_MAX
    );
  }
  for(int i=0;i<num_bins;i++)
  {
    h_bin_counters[i] = 0; h_bin_counters_checker[i] = 0;
  }
  for(int i=0;i<num_bins*bin_size;i++)
  {
    h_bins[i] = -1;
    h_bins_checker[i] = -1;
  }

  for(int i=0;i<num_particles*NUM_NEIGHBORS;i++)
  {
    h_knn[i] = -1;
    h_knn_checker[i] = -1;
  }

  cudaMemcpy(
    d_particles, h_particles, num_particles * sizeof(float3),
    cudaMemcpyHostToDevice
  );
  check_cuda_error("Memcpy error");

  cudaMemset(d_bins, -1, num_bins * bin_size * sizeof(int));
  cudaMemset(d_knn, -1, NUM_NEIGHBORS * num_particles * sizeof(int));
  cudaMemset(d_bin_counters, 0, num_bins * sizeof(int));
  check_cuda_error("Memset error");

  start_timer(&timer);
  device_binning<<<num_particles / 256, 256>>>(
    d_particles, d_bins, d_bin_counters, gridding, bin_size, num_particles
  );
  check_cuda_error("Binning error");
  stop_timer(&timer,"Host binning completed");

  const dim3 blockSize(
    4,
    16,
    16
  );

  start_timer(&timer);
  device_binned_knn<<<num_bins/1024, blockSize>>>(
    d_particles, d_bins, d_knn, binning_dim, bin_size
  );
  check_cuda_error("Binned knn error");
  stop_timer(&timer,"Device binned knn completed");

  cudaMemcpy(
    h_bin_counters, d_bin_counters, num_bins * sizeof(int),
    cudaMemcpyDeviceToHost
  );
  cudaMemcpy(
    h_knn, d_knn, NUM_NEIGHBORS * num_particles * sizeof(int),
    cudaMemcpyDeviceToHost
  );

  // generate reference output
  start_timer(&timer);
  host_binning(
    h_particles_checker, h_bins_checker, h_bin_counters_checker, gridding,
    bin_size, num_particles
  );
  stop_timer(&timer,"Host binning completed");

  start_timer(&timer);
  host_binned_knn(
    h_particles_checker, h_bins_checker, h_knn_checker, binning_dim, bin_size
  );
  stop_timer(&timer,"Host binned knn completed");

  // check CUDA output versus reference output
  cross_check_results(
    num_particles, num_bins, bin_size, h_bin_counters,
    h_bin_counters_checker, h_knn, h_knn_checker
  );
  deallocate_host_memory(
    h_particles, h_bins, h_knn, h_bin_counters, h_particles_checker,
    h_bins_checker, h_knn_checker, h_bin_counters_checker
  );

  deallocate_device_memory(d_particles, d_bins, d_knn, d_bin_counters);

  return 0;
}
