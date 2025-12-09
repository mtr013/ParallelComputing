#include "com2039.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#define BLOCK_SIZE 256

// function to load samples from file
size_t loadSamples(std::string fileName, float **addressToSamples) {
    std::string line;
    std::ifstream inputFile(fileName);
    if (!inputFile.is_open()) {
        std::cerr << "error: unable to open file " << fileName << std::endl;
        exit(EXIT_FAILURE);
    }

    // count number of lines to allocate right amount of memory
    size_t numLines = 0;
    while (getline(inputFile, line)) {
        numLines++;
    }

    // allocate memory on CPU
    *addressToSamples = new float[numLines];

    // return to beginning of the file and fill in data
    inputFile.clear();
    inputFile.seekg(0);
    for (int j = 0; j < numLines; j++) {
        getline(inputFile, line);
        (*addressToSamples)[j] = std::stof(line);
    }

    inputFile.close();
    return numLines;
}

/////// find maximum
/*
 * 	I developed a parallel reduction method to get the maximum value of the input sequence.
 * 	maxReduceKernel compares elements in each block
 * 	and updates the maximum value with -INFINITY as identity element.
 * 	Threads keep reducing the data until only 1 maximum value is there.
 * 	Then, I allocated memory for input data on the GPU.
 */
__global__ void maxReduceKernel(float* d_in, int lenArray) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // load data into shared memory
    if (i < lenArray) {
        sdata[tid] = d_in[i];
    } else {
        // identity element (-INFINITY) for out-of-bounds elements
        sdata[tid] = -INFINITY;
    }
    __syncthreads();

    // reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // write result back to global memory
    if (tid == 0) d_in[blockIdx.x] = sdata[0];
}

float findMaxValue(float* samples_h, size_t numSamples) {
    // allocate memory on GPU
    float* d_samples;
    cudaMalloc(&d_samples, numSamples * sizeof(float));
    cudaMemcpy(d_samples, samples_h, numSamples * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (numSamples + threadsPerBlock - 1) / threadsPerBlock;

    // call kernel iteratively until only one element remains
    while (numSamples > 1) {
        maxReduceKernel<<<blocksPerGrid, threadsPerBlock, BLOCK_SIZE * sizeof(float)>>>(d_samples, numSamples);
        numSamples = (numSamples + threadsPerBlock - 1) / threadsPerBlock;
    }

    // copy result back to host
    float max_val;
    cudaMemcpy(&max_val, d_samples, sizeof(float), cudaMemcpyDeviceToHost);

    // free GPU memory
    cudaFree(d_samples);
    return max_val;
}

/////// find minimum
/*
 * 	I developed a parallel reduction method to get the minimum value of the input sequence.
 * 	minReduceKernel compares elements in each block
 * 	and updates the minimum value with INFINITY as identity element.
 * 	Threads keep reducing the data until only 1 minimum value is there.
 *  Then, I allocated memory for input data on the GPU.
 */
__global__ void minReduceKernel(float* d_in, int lenArray) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // load data into shared memory
    if (i < lenArray) {
        sdata[tid] = d_in[i];
    } else {
        // identity element (INFINITY) for out-of-bounds elements
        sdata[tid] = INFINITY;
    }
    __syncthreads();

    // reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // write result back to global memory
    if (tid == 0) d_in[blockIdx.x] = sdata[0];
}

float findMinValue(float* samples_h, size_t numSamples) {
    // allocate memory on GPU
    float* d_samples;
    cudaMalloc(&d_samples, numSamples * sizeof(float));
    cudaMemcpy(d_samples, samples_h, numSamples * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (numSamples + threadsPerBlock - 1) / threadsPerBlock;

    // call kernel iteratively until only one element remains
    while (numSamples > 1) {
        minReduceKernel<<<blocksPerGrid, threadsPerBlock, BLOCK_SIZE * sizeof(float)>>>(d_samples, numSamples);
        numSamples = (numSamples + threadsPerBlock - 1) / threadsPerBlock;
    }

    // copy result back to host
    float min_val;
    cudaMemcpy(&min_val, d_samples, sizeof(float), cudaMemcpyDeviceToHost);

    // free GPU memory
    cudaFree(d_samples);
    return min_val;
}

/////// create histogram
/*
 *  I made a parallel histogram function
 *  that correctly assigns points to individual bins (0 to 255).
 *  histogramKernel256 calculates the bin index
 *  for each input and updates the respective bin count.
 *  It divides the range of values into 256 bins
 *  and makes sure each element is assigned to the right bin.
 */
__global__ void histogramKernel256(float* d_in, unsigned int* hist, size_t lenArray, float minValue, float maxValue) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < lenArray) {
        float bin_width = (maxValue - minValue) / 256.0f;
        int bin = min(255, (int)floor((d_in[tid] - minValue) / bin_width));

        // atomic operation to update histogram bins
        /*
         *  I used atomicAdd to prevent race conditions
         *  when multiple threads update the same histogram bin at the same time
         *  by serialising write operations and making the data consistent.
         *  Without it, the histogram counts would be incorrect because of concurrent writes.
         */
        atomicAdd(&hist[bin], 1);
    }
}

/// histogram
void histogram256(float* samples_h, size_t numSamples, unsigned int** hist_h, float minValue, float maxValue) {
    // allocate memory for histogram on GPU
    unsigned int* d_hist;
    cudaMalloc(&d_hist, 256 * sizeof(unsigned int));
    cudaMemset(d_hist, 0, 256 * sizeof(unsigned int));

    // allocate memory for samples on GPU
    float* d_samples;
    cudaMalloc(&d_samples, numSamples * sizeof(float));
    cudaMemcpy(d_samples, samples_h, numSamples * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (numSamples + threadsPerBlock - 1) / threadsPerBlock;

    // call histogram kernel
    histogramKernel256<<<blocksPerGrid, threadsPerBlock>>>(d_samples, d_hist, numSamples, minValue, maxValue);

    // allocate pinned memory for histogram on CPU
    cudaMallocHost((void**)hist_h, 256 * sizeof(unsigned int));
    cudaMemcpy(*hist_h, d_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // free GPU memory
    cudaFree(d_samples);
    cudaFree(d_hist);
}
