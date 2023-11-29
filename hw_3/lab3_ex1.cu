
#include <stdio.h>
#include <time.h>
#include <random>

#define NUM_BINS 4096
#define THREADS_PER_BLOCK 128

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins)
{

  //@@ Insert code below to compute histogram of input using shared memory and atomics
  __shared__ unsigned int hist[NUM_BINS];
  int id = threadIdx.x + blockDim.x * blockIdx.x;

  if (threadIdx.x < NUM_BINS)
  {
    hist[threadIdx.x] = 0;
  }

  __syncthreads();

  if (id < num_elements)
  {
    atomicAdd(&(hist[input[id]]), 1);
  }

  __syncthreads();

  if (threadIdx.x < NUM_BINS)
  {
    // threadIdx.x = 0..THREADS_PER_BLOCK
    // i = 0..threadIdx.x * THREADS_PER_BLOCK
    for (int i = threadIdx.x * THREADS_PER_BLOCK; i < (threadIdx.x + 1) * THREADS_PER_BLOCK; i++)
    {
      atomicAdd(&(bins[i]), hist[threadIdx.x]);
    }
  }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins)
{

  //@@ Insert code below to clean up bins that saturate at 127
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id < num_bins)
  {
    if (bins[id] > 127)
    {
      bins[id] = 127;
    }
  }
}

void createSaturatedHistogram(unsigned int *input, unsigned int *bins,
                              unsigned int num_elements)
{
  for (int i = 0; i < NUM_BINS; i++)
  {
    bins[i] = 0;
  }

  int max = 0;
  for (int i = 0; i < num_elements; i++)
  {
    if (input[i] > max)
    {
      max = input[i];
    }
  }

  int binSize = std::max(max / NUM_BINS, 1);
  for (int i = 0; i < num_elements; i++)
  {
    int bin = input[i] / binSize;
    if (bin > NUM_BINS - 1)
    {
      bin = NUM_BINS - 1;
    }
    bins[bin]++;
  }

  for (int i = 0; i < NUM_BINS; i++)
  {
    if (bins[i] > 127)
    {
      bins[i] = 127;
    }
  }
}

int main(int argc, char **argv)
{

  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args
  if (argc > 1)
  {
    inputLength = atoi(argv[1]);
  }
  else
  {
    inputLength = 1;
  }

  //@@ Insert code below to allocate Host memory for input and output
  hostInput = (unsigned int *)malloc(inputLength * sizeof(unsigned int));
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));

  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  srand(time(NULL));
  for (int i = 0; i < inputLength; i++)
  {
    hostInput[i] = rand() % NUM_BINS;
  }

  for (int i = 0; i < inputLength; i++)
  {
    printf("%d\n", hostInput[i]);
  }

  //@@ Insert code below to create reference result in CPU
  resultRef = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  // unsigned int test[10] = {1, 1, 1, 1, 1, 5, 5, 5, 5, 5};
  createSaturatedHistogram(hostInput, resultRef, inputLength);
  printf("The reference result is: \n");
  for (int i = 0; i < NUM_BINS; i++)
  {
    if (resultRef[i] != 0)
      printf("%d: %d\n", i, resultRef[i]);
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int));
  cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int));

  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceBins, hostBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

  //@@ Initialize the grid and block dimensions here
  int numBlocks = (inputLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  //@@ Launch the GPU Kernel here
  histogram_kernel<<<numBlocks, THREADS_PER_BLOCK>>>(deviceInput, deviceBins, inputLength, NUM_BINS);

  //@@ Initialize the second grid and block dimensions here
  int numBlocks2 = (NUM_BINS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  //@@ Launch the second GPU Kernel here
  // convert_kernel<<<numBlocks2, THREADS_PER_BLOCK>>>(deviceBins, NUM_BINS);

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  printf("The GPU result is: \n");
  for (int i = 0; i < NUM_BINS; i++)
  {
    if (hostBins[i] != 0)
      printf("%d: %d\n", i, hostBins[i]);
  }

  //@@ Insert code below to compare the output with the reference
  for (int i = 0; i < NUM_BINS; i++)
  {
    if (hostBins[i] != resultRef[i])
    {
      printf("The result is wrong!\n");
      printf("index %d: %d %d\n", i, hostBins[i], resultRef[i]);
      return 1;
    }
  }

  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);

  //@@ Free the CPU memory here
  free(hostInput);
  free(hostBins);
  free(resultRef);

  return 0;
}
