
#include <stdio.h>
#include <sys/time.h>
#include <string>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
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


int main(int argc, char **argv) {
  
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]);
  
  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (DataType *) malloc(inputLength * sizeof(DataType));
  hostInput2 = (DataType *) malloc(inputLength * sizeof(DataType));
  hostOutput = (DataType *) malloc(inputLength * sizeof(DataType));
  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  srand(time(NULL));
  resultRef = (DataType *) malloc(inputLength * sizeof(DataType));
  for (int i = 0; i < inputLength; i++) {
    hostInput1[i] = rand() % 100; 
    hostInput2[i] = rand() % 100;
    resultRef[i] = hostInput1[i] + hostInput2[i];
  }

  for (int i = 0; i < inputLength; i++) {
    resultRef[i] = hostInput1[i] + hostInput2[i];
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
  cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
  cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));


  //@@ Insert code to below to Copy memory to the GPU here
  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);

  //@@ Initialize the 1D grid and block dimensions here

  // https://stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels
  // CUDA 6.5 has runtime functions that aid in calculating the maxmium occupancy
  int blockSize;
  int minGridSize;
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vecAdd, 0, inputLength);
  int gridSize = (inputLength + blockSize - 1) / blockSize;

  //@@ Launch the GPU Kernel here
  vecAdd<<<gridSize, blockSize>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);


  //@@ Insert code below to compare the output with the reference
  for (int i = 0; i < inputLength; i++) {
    //using 1e-6 as epsilon for comparison
    if (abs(hostOutput[i] - resultRef[i]) > 1e-6) {
      printf("Error: hostOutput[%d] = %f, resultRef[%d] = %f\n", i, hostOutput[i], i, resultRef[i]);
      return 1;
    } 
  }

  printf("Success!\n");

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
