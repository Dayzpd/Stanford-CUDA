__host__ __device__ unsigned int bin_index(float3 particle, int3 gridding)
{
  unsigned int x_index = (unsigned int)(particle.x * (1 << gridding.x));
  unsigned int y_index = (unsigned int)(particle.y * (1 << gridding.y));
  unsigned int z_index = (unsigned int)(particle.z * (1 << gridding.z));
  unsigned int index = 0;
  index |= z_index;
  index <<= gridding.y;
  index |= y_index;
  index <<= gridding.x;
  index |=  x_index;

  return index;
}

__host__ __device__
float dist2(float3 a, float3 b)
{
  float3 d = a - b;
  float d2 = d.x*d.x + d.y*d.y + d.z*d.z;
  return d2;
}

template
<typename T>
__host__ __device__
void init_list(T *base_ptr, unsigned int size, T val)
{
  for(int i=0;i<size;i++)
  {
    base_ptr[i] = val;
  }
}

__host__ __device__
void insert_list(float *dist_list, int *id_list, float dist, int id)
{
 int k;
 for (k=0; k < NUM_NEIGHBORS; k++) {
   if (dist < dist_list[k]) {
     // we should insert it in here, so push back and make it happen
     for (int j = NUM_NEIGHBORS - 1; j > k ; j--) {
       dist_list[j] = dist_list[j-1];
       id_list[j] = id_list[j-1];
     }
     dist_list[k] = dist;
     id_list[k] = id;
     break;
   }
 }
}

bool cross_check_results(
  int num_particles, int num_bins, int bin_size, int *h_bin_counters,
  int *h_bin_counters_checker, int *h_knn, int *h_knn_checker
) {
  int error = 0;

  for(int i=0;i<num_bins;i++)
  {
    if(h_bin_counters[i] != h_bin_counters_checker[i])
    {
      printf(
        "mismatch! bin %d: cuda:%d host:%d particles \n",
        i,
        h_bin_counters[i],
        h_bin_counters_checker[i]
      );

      error = 1;
    }
  }
  for(int i=0;i<num_particles;i++)
  {
    for(int j=0;j<NUM_NEIGHBORS;j++)
    {
      if(h_knn[i*NUM_NEIGHBORS + j] != h_knn_checker[i*NUM_NEIGHBORS + j])
      {
      printf(
          "mismatch! particle: %d, neighbor %d, d_knn %d: h_knn:%d \n",
          i,
          j,
          h_knn[i*NUM_NEIGHBORS + j],
          h_knn_checker[i*NUM_NEIGHBORS + j]
        );
        error = 1;
      }
    }
  }

  if(error)
  {
  printf("Output of CUDA version and normal version didn't match! \n");
  }
  else {
  printf("Worked! CUDA and reference output match. \n");
  }
  return error;
}

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %zd\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %zd\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %zd\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %zd\n",  devProp.totalConstMem);
    printf("Texture alignment:             %zd\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}
