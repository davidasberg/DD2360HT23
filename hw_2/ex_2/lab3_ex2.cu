#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#define DataType float
#define TPB 16

__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here
  
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if ((i >= numARows) ||(j >= numBColumns) ) return;
  DataType temp_sum = 0.0;
  for(int k=0; k<numBRows; k++){
    temp_sum += A[i*numAColumns+k] * B[k*numBColumns+j];
  }
  C[i*numBColumns+j] = temp_sum;
}

DataType *matrix_multiplication(DataType *A, DataType *B, DataType *C, int numARows,
  int numAColumns, int numBRows, int numBColumns) {
  
    for (int i = 0; i < numARows; i++) {
        for (int k = 0; k < numBRows; k++) {
            for (int j = 0; j < numBColumns; j++) {
                C[i * numBColumns + j] += A[i * numAColumns + k] * B[k * numBColumns + j];
            }
        }
    }
  
    return C;
}

int main(int argc, char **argv) {
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  std::chrono::time_point<std::chrono::system_clock> start_copy_to_device, end_copy_to_device, start_kernel, end_kernel, start_copy_to_host, end_copy_to_host;


  //if length of args is less than 4, print error message
  if(argc < 4){
    printf("Usage: ./matrixmul <numARows> <numAColumns> <numBColumns>\n");
    return 1;
  }

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
  numARows = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBRows = numAColumns;
  numBColumns = atoi(argv[3]);
  numCRows = numARows;
  numCColumns = numBColumns;
  
  //@@ Insert code below to allocate Host memory for input and output
  hostA = (DataType *)malloc(numARows * numAColumns * sizeof(DataType));
  hostB = (DataType *)malloc(numBRows * numBColumns * sizeof(DataType));
  hostC = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));
  
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  srand(time(NULL));
  for(int i=0; i<numARows; i++){
    for(int j=0; j<numAColumns; j++){
      hostA[i*numAColumns+j] = (DataType)rand() / (DataType)RAND_MAX;
    }
  }

  for(int i=0; i<numBRows; i++){
    for(int j=0; j<numBColumns; j++){
      hostB[i*numBColumns+j] = (DataType)rand() / (DataType)RAND_MAX;
    }
  }


  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceA, numARows * numAColumns * sizeof(DataType));
  cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(DataType));
  cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here
  start_copy_to_device = std::chrono::system_clock::now();
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  end_copy_to_device = std::chrono::system_clock::now();

  //@@ Initialize the grid and block dimensions here

  int d_grid_x = (numARows + TPB - 1) / TPB; 
  int d_grid_y = (numBColumns + TPB - 1) / TPB;

  //@@ Launch the GPU Kernel here
  start_kernel = std::chrono::system_clock::now();
  gemm<<<dim3(d_grid_x, d_grid_y, 1), dim3(TPB, TPB, 1)>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();
  end_kernel = std::chrono::system_clock::now();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
    printf("Error: %s\n", cudaGetErrorString(err));
  
  //@@ Copy the GPU memory back to the CPU here
  start_copy_to_host = std::chrono::system_clock::now();
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost);
  end_copy_to_host = std::chrono::system_clock::now();

  //@@ Insert code below to compare the output with the reference
  resultRef = matrix_multiplication(hostA, hostB, hostC, numARows, numAColumns, numBRows, numBColumns);
  bool match = true;

  for(int i=0; i<numARows; i++){
    for(int j=0; j<numBColumns; j++){
      if(hostC[i*numBColumns+j] != resultRef[i*numBColumns+j]){
        match = false;
        break;
      }
    }
  }

  if(match) printf("Result matches\n");
  else printf("Result does not match\n");
  
  std::chrono::duration<double> copy_to_device = end_copy_to_device - start_copy_to_device;
  std::chrono::duration<double> kernel = end_kernel - start_kernel;
  std::chrono::duration<double> copy_to_host = end_copy_to_host - start_copy_to_host;

  // multiply by 1000 to convert to milliseconds
  printf("Copy to device: %f\n", copy_to_device.count() * 1000);
  printf("Kernel: %f\n", kernel.count() * 1000);
  printf("Copy to host: %f\n", copy_to_host.count() * 1000);


  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  // //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
