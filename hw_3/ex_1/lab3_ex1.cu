
#include <stdio.h>
#include <random>
#include <time.h>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NUM_BINS 4096
#define THREADS_PER_BLOCK 128

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins)
{

    //@@ Insert code below to compute histogram of input using shared memory and atomics
    __shared__ unsigned int hist[NUM_BINS];
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    int stride = blockDim.x;
    for (int i = threadIdx.x; i < NUM_BINS; i += stride)
    {
        hist[i] = 0;
    }

    __syncthreads();

    if (id < num_elements)
    {
        atomicAdd(&(hist[input[id]]), 1);
    }

    __syncthreads();

    // let one thread do the work
    // if (threadIdx.x == 0)
    // {
    //     for (int i = 0; i < NUM_BINS; i++)
    //     {
    //         atomicAdd(&(bins[i]), hist[i]);
    //     }
    // }

    // divide the work among threads
    for (int i = threadIdx.x; i < NUM_BINS; i += stride)
    {
        atomicAdd(&(bins[i]), hist[i]);
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
    // Set all bins to 0
    for (int i = 0; i < NUM_BINS; i++)
    {
        bins[i] = 0;
    }
    // Iterate through all input bins
    for (int i = 0; i < num_elements; i++)
    {
        int bin = input[i];
        if (bin > NUM_BINS - 1)
        {
            bin = NUM_BINS - 1;
        }
        bins[bin]++;
    }
    // Saturate bins at 127
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

    std::chrono::time_point<std::chrono::system_clock> start_copy_to_device, end_copy_to_device, start_hist_kernel, end_hist_kernel, start_convert_kernel, end_convert_kernel, start_copy_to_host, end_copy_to_host;

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
        hostInput[i] = 4095; // rand() % NUM_BINS;
    }

    // for (int i = 0; i < inputLength; i++)
    // {
    //     printf("%d\n", hostInput[i]);
    // }

    //@@ Insert code below to create reference result in CPU
    resultRef = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
    // unsigned int test[10] = {1, 1, 1, 1, 1, 5, 5, 5, 5, 5};
    createSaturatedHistogram(hostInput, resultRef, inputLength);
    // printf("The reference result is: \n");
    // for (int i = 0; i < NUM_BINS; i++)
    // {
    //     if (resultRef[i] != 0)
    //         printf("%d: %d\n", i, resultRef[i]);
    // }

    //@@ Insert code below to allocate GPU memory here
    cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int));
    cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int));

    //@@ Insert code to Copy memory to the GPU here
    start_copy_to_device = std::chrono::system_clock::now();
    cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBins, hostBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyHostToDevice);
    end_copy_to_device = std::chrono::system_clock::now();

    //@@ Insert code to initialize GPU results
    cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

    //@@ Initialize the grid and block dimensions here
    int numBlocks = (inputLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    //@@ Launch the GPU Kernel here
    start_hist_kernel = std::chrono::system_clock::now();
    histogram_kernel<<<numBlocks, THREADS_PER_BLOCK>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
    end_hist_kernel = std::chrono::system_clock::now();

    //@@ Initialize the second grid and block dimensions here
    int numBlocks2 = (NUM_BINS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    //@@ Launch the second GPU Kernel here
    start_convert_kernel = std::chrono::system_clock::now();
    convert_kernel<<<numBlocks2, THREADS_PER_BLOCK>>>(deviceBins, NUM_BINS);
    end_convert_kernel = std::chrono::system_clock::now();

    //@@ Copy the GPU memory back to the CPU here
    start_copy_to_host = std::chrono::system_clock::now();
    cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    end_copy_to_host = std::chrono::system_clock::now();

    printf("The GPU result is: \n");
    // for (int i = 0; i < NUM_BINS; i++)
    // {
    //     if (hostBins[i] != 0)
    //         printf("%d: %d\n", i, hostBins[i]);
    // }

    //@@ Insert code below to compare the output with the reference
    bool isCorrect = true;
    for (int i = 0; i < NUM_BINS; i++)
    {
        if (hostBins[i] != resultRef[i])
        {
            printf("The result is wrong!\n");
            printf("index %d: %d %d\n", i, hostBins[i], resultRef[i]);
            isCorrect = false;
        }
    }
    if (!isCorrect)
    {
        return 1;
    }

    printf("The result is correct!\n");
    printf("Copy to device: %f ms\n", std::chrono::duration<double, std::milli>(end_copy_to_device - start_copy_to_device).count());
    printf("Histogram kernel: %f ms\n", std::chrono::duration<double, std::milli>(end_hist_kernel - start_hist_kernel).count());
    printf("Convert kernel: %f ms\n", std::chrono::duration<double, std::milli>(end_convert_kernel - start_convert_kernel).count());
    printf("Copy to host: %f ms\n", std::chrono::duration<double, std::milli>(end_copy_to_host - start_copy_to_host).count());

    //@@ Free the GPU memory here
    cudaFree(deviceInput);
    cudaFree(deviceBins);

    //@@ Free the CPU memory here
    free(hostInput);
    free(hostBins);
    free(resultRef);

    return 0;
}
