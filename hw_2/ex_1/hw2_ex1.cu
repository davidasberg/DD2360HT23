
#include <stdio.h>
#include <time.h>
#include <string>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define DataType float
#define THREADS_PER_BLOCK 1024

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len)
{
  //@@ Insert code to implement vector addition here
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len)
  {
    out[i] = in1[i] + in2[i];
  }
}

//@@ Insert code to implement timer start
// void timerStart(struct timeval *start) {
//   gettimeofday(start, NULL);
// }

//@@ Insert code to implement timer stop
// void timerStop(struct timeval *start, std::string label) {
//   struct timeval stop;
//   gettimeofday(&stop, NULL);
//   printf("%s: %lu us\n", label.c_str(), (stop.tv_sec - start->tv_sec) * 1000000 + (stop.tv_usec - start->tv_usec));
// }

int main(int argc, char **argv)
{

  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  std::chrono::time_point<std::chrono::system_clock> start_copy_to_device, end_copy_to_device, start_kernel, end_kernel, start_copy_to_host, end_copy_to_host;

  //@@ Insert code below to read in inputLength from args
  if (argc < 2)
  {
    printf("Usage: ./vecAdd inputLength\n");
    return 1;
  }
  inputLength = atoi(argv[1]);

  printf("The input length is %d\n", inputLength);

  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (DataType *)malloc(inputLength * sizeof(DataType));
  hostInput2 = (DataType *)malloc(inputLength * sizeof(DataType));
  hostOutput = (DataType *)malloc(inputLength * sizeof(DataType));

  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  srand(time(NULL));
  resultRef = (DataType *)malloc(inputLength * sizeof(DataType));
  for (int i = 0; i < inputLength; i++)
  {
    hostInput1[i] = rand() % 100;
    hostInput2[i] = rand() % 100;
    resultRef[i] = hostInput1[i] + hostInput2[i];
  }

  for (int i = 0; i < inputLength; i++)
  {
    resultRef[i] = hostInput1[i] + hostInput2[i];
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
  cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
  cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here
  start_copy_to_device = std::chrono::system_clock::now();
  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  end_copy_to_device = std::chrono::system_clock::now();

  //@@ Initialize the 1D grid and block dimensions here

  // https://stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels
  // CUDA 6.5 has runtime functions that aid in calculating the maxmium occupancy
  // int blockSize;
  // int minGridSize;
  // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vecAdd, 0, inputLength);
  int blockSize = THREADS_PER_BLOCK;
  int gridSize = (inputLength + blockSize - 1) / blockSize;

  //@@ Launch the GPU Kernel here
  start_kernel = std::chrono::system_clock::now();
  vecAdd<<<gridSize, blockSize>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  end_kernel = std::chrono::system_clock::now();

  //@@ Copy the GPU memory back to the CPU here
  start_copy_to_host = std::chrono::system_clock::now();
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);
  end_copy_to_host = std::chrono::system_clock::now();

  //@@ Insert code below to compare the output with the reference
  for (int i = 0; i < inputLength; i++)
  {
    // using 1e-6 as epsilon for comparison
    if (abs(hostOutput[i] - resultRef[i]) > 1e-6)
    {
      printf("Error: hostOutput[%d] = %f, resultRef[%d] = %f\n", i, hostOutput[i], i, resultRef[i]);
      return 1;
    }
  }

  printf("Success!\n");
  printf("Copy to device: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end_copy_to_device - start_copy_to_device).count() / 1000.0);
  printf("Kernel: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end_kernel - start_kernel).count() / 1000.0);
  printf("Copy to host: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end_copy_to_host - start_copy_to_host).count() / 1000.0);

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);

  return 0;
}
