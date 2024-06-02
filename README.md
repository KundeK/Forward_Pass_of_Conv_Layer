# Project Summary

The goal of this project was to reduce the execution time of the forward pass for improved real-time processing and throughput, particularly for applications such as autonomous vehicles, augmented reality, and faster decision-making in response to environmental changes.

## CPU Implementation

### Convolution Function

**Purpose:**
- Understand the basics of unoptimized convolution and identify bottlenecks using `gprof`.

**Function Parameters:**
- **output:** Destination for the convolutional layer's output feature maps.
- **input:** The input images.
- **mask:** The convolutional filters or kernels.
- **Batch:** The batch size, indicating the number of images processed together.
- **Map_out:** Number of output feature maps per input image.
- **Channel:** Number of channels in each input image.
- **Height, Width:** Dimensions of the input images.
- **K:** Dimension of the square convolutional kernel (K x K).

### Process Overview

1. **Output Dimensions Calculation:** Determine dimensions of the output feature maps based on input sizes and kernel dimensions.
2. **Simplified Indexing:** Use macros (out_4d, in_4d, mask_4d) for easy access to elements within the multi-dimensional arrays.
3. **Convolution Operation:**
   - Iterate over each image, output feature map, and pixel location in the output feature map.
   - Initialize each output pixel to 0, compute its value by iterating over each input channel and kernel element, summing products of corresponding input pixels and kernel values.
   - Repeat for each pixel in the output feature maps, completing the convolution for each image in the batch.

### Performance Analysis using gprof

- **Function Call Counts and Times:**
  - Number of times each function is called.
  - Execution time, including total time and self time (excluding called functions).

- **Call Graph Information:**
  - Maps function calls to identify bottlenecks, either due to inherent slowness or frequent calls.

## Transition to GPU

Implemented a basic CUDA kernel for convolution and performed performance analysis using `nsys` and `Nsight Compute`.

### nsys Profile Data

- **CUDA API Statistics:** Lists CUDA API calls and their time cost. High time percentages suggest optimization opportunities (e.g., `cudaMalloc` and `cudaMemcpy`).
- **CUDA Kernel Statistics:** Shows information about CUDA kernels. Focus on optimizing time-consuming kernels.
- **CUDA Memory Operation Statistics:** Details on memory operations, both in time and bytes transferred. Consider reducing memory traffic or optimizing memory access patterns.

### Nsight Compute Profile Data

- **GPU Utilization:** Indicates GPU resource usage. Aim for high utilization for maximum performance.
- **Memory Throughput:** Shows data read/write rates to GPU memory. Low throughput suggests optimizing memory access patterns.
- **L1/TEX Cache and L2 Hit Rates:** Low hit rates indicate inefficient memory access patterns, leading to increased global memory traffic.
- **Shared Memory:** Indicates shared memory usage by kernels. Efficient use can significantly impact performance.

## Optimizations

### Tiled Shared Memory Convolution

- Divides data into tiles stored in shared memory where convolution is performed simultaneously.
- Reduces memory bandwidth bottleneck by improving cache utilization and taking advantage of parallelism.

### Loop Unrolling

- Allows compiler to optimize memory accesses, improving cache utilization and reducing memory access latency.
- Increases task parallelism, boosting performance.

### Streams to Overlap Computation

- Divides device resources into different streams for operations like data transfer or convolution.
- Streams work concurrently, reducing the time needed for data transfer as it happens simultaneously with convolution.

## Results

- Achieved an accuracy rate of 87.14% by optimizing the code.
- Reduced execution time from 170ms to 100ms using tiled shared memory convolution and loop unrolling techniques.

---

For more detailed information and code, refer to the source files and comments within the project repository.

