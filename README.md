
# Scientific computing with Metal in C++: Matrix multiplication example



This repo currently contains example Metal shaders (kernels) for performing matrix multiplication on the GPU. One of these kernels was ported from an existing CUDA kernel and so it also serves as a concrete example of porting existing CUDA code to Metal. Code is included for measuring the performance of the shaders. Finally, some preliminary experiments are performed to explore the potential for simplified GPU programming and increased performance due to the physically shared CPU-GPU memory of M1-based Macs.

### Background and motivation

I wrote this because I was curious to see how the performance of simple custom Metal shaders (i.e., kernels) compare to CUDA. I was also curious to see if the physical shared CPU-GPU memory of Apple's M1-based Macs could potentially offer some benefits in terms of performance and/or ease of development compared to existing GPUs with their physically separate memory spaces. 

I initially thought a reasonable starting point would be to find and experiment with some simple Metal matrix multiplication examples similar to the naive and tiled (i.e., using threadlocal GPU memory) kernels that are often covered in introductory CUDA tutorials. Since Apple has released [Metal-cpp](https://developer.apple.com/metal/cpp/) to enable C++ development with Metal, I searched for some example matrix multiplication shaders. Unfortunately, as of August 2022, it seemed that there were none.

The most related existing resource I could find were the following:
- Apple's developer site has an article with sample code: [Performing Calculations on a GPU](https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu). I used this as a starting point but the example is for element-wise addition only and not C++ (they used Swift and Objective-C).
- Apple's [Metal Sample Code](https://developer.apple.com/metal/sample-code/) contains one C++ example, but it and the other Swift examples are related to graphics rendering.
- The sole existing example of someone providing example C++ and Metal code for scientific computing seems to be Lars Gebraad, who has a repo [here](https://github.com/larsgeb/m1-gpu-cpp) and a corresponding paper [here](https://arxiv.org/abs/2206.01791) where in addition to porting Apple's introductory array adder to C++, he also provides other examples such as partial differential equations. In addition, his performance comparisions found Metal on M1-based Macs to be promising, often providing significant speedups compared to CPU code.

Since there seemed to be no existing open source matrix multiplication shaders in Metal, I decided to write my own. Since I am relatively new to Metal, I thought it would also be a good learning excercise. These shaders are intended to serve as examples only. If you are only interested in getting the best possible matrix multiplication performance, then I would suggest using an existing (proprietary) BLAS implementation instead such as Accelerate's `sgemm` for CPU/AMX or `MPSMatrixMultiplication` for GPU (although it only seems to be available for Swift and Objective-C at the moment).


### Description of the included code

This repo includes two matrix multiplication shaders:
- `mat_mul_simple1.metal`: The most basic GPU implementation possible. This is essentially the same code as the inner loop of the CPU version. It simply computes the dot product over the inner dimension for its current thread. We will refer to this as the "naive" GPU implementation. Arbitrary matrix sizes are supported.
- `mat_mul_optimized_nv.metal`: This version uses shared threadgroup memory with a tiled algorithm. I directly ported it to Metal from the corresponding CUDA kernel in NVIDIA's [cuda-samples](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/matrixMul/matrixMul.cu). Arbitrary matrix sizes are supported. (note that the original CUDA kernel I linked to did *not* support arbitrary matrix sizes, so I modified it slightly while porting)

I also include the `Matrix` tensor (multidimensional array) class and a subset of the tensor `Utilities` from my old Kumozu framework. It contains CPU implementations of matrix multiplication including BLAS and some other simple tensor operations. This is only included because it simplified writing the examples. In your own code, you could replace this with your favorite tensoor library, provided that it is also capable of initilizing a tensor given an externally supplied float pointer.

The `MatrixMultiplier` class takes one of the above kernel names, initialized the shared memory matrices to be multiplied, and can then perform either CPU or GPU matrix multiplication.

The `main.cpp` file contains a function to run each experiment, which performs the benchmarking.

Note: This repo uses C++ and Metal only. If you are simply looking for Metal matrix multiplication shader examples to use in your Swift code, I suspect these will work (i.e., the .metal files), but you will then need to write your own Swift code to call them.

-----------
### Install and setup

This software is intended to be run on an Apple Silicon-based Mac. It was developed on a 2021 Macbook Pro with M1 Max.

Install Xcode.

Clone this repo and open the `.xcodeproj` project file in Xcode.

It should then build and run. (although I have not tested that it works on another machine)

-----------
### Performance results and notes

The following results are for a 2021 Macbook Pro with M1 Max and 64 GB RAM.
The code was run in release mode with clang optimization flags `-O3 -ffast-math`.

#### Experiment 1: Performance of the naive shader

Let's first consider the most basic "naive" matrix multiplication implementation on CPU and GPU and compare it to an optimized CPU sgemm in Apple's Accelerate Framework. It will compute "X = A * B".

The code in function `run_mat_mult_shaders()` in `main.cpp` actually runs both the naive shader (`shader_name = "mat_mul_simple1"`, which corresponds to the shader in `mat_mul_simple1.metal`) and the optimized shader discussed in the next section.


You can change the matrix dimensions and rebuild to try different sizes. You can also experiment with different threadgroup sizes. 8x8 was used for the experiments unless otherwise specified. Larger values up to 32x32 are possible and could perform better in some cases. If chaning it, note that the value appears in two places in the code (in the .metal source and the .cpp source that calls it) so be sure to update both.

Let's benchmark the performance of the operation "X = A * B" for a few different matrix sizes. I get the following results:

| Implementation | A size | B size | GFLOPS |
|----------------|--------|--------|--------|
| `mat_mul_simple1` | 256x256 | 256x256 | 76.4 |
| Naive CPU | 256x256 | 256x256 |2.17 |
| Accelerate BLAS | 256x256 | 256x256 | 1097.9 |
| | | | |
| `mat_mul_simple1` | 1230x789 | 789x1156 | 583.8 |
| Naive CPU | 1230x789 | 789x1156 | 1.74 |
| Accelerate BLAS | 1230x789 | 789x1156 | 1484.8 |
| | | | |
| `mat_mul_simple1` | 1024x1024 | 1024x1024 | 578.3 |
| Naive CPU | 1024x1024 | 1024x1024 | 1.70 |
| Accelerate BLAS | 1024x1024 | 1024x1024 | 1724.8 |
| | | | |
| `mat_mul_simple1` | 2048x2048 | 2048x2048 | 652.5 |
| Naive CPU | 2048x2048 | 2048x2048 | 0.74 |
| Accelerate BLAS | 2048x2048 | 2048x2048 | 1883.4 |
| | | | |
| `mat_mul_simple1` | 3000x5000 | 5000x4000 | 553.8 |
| Naive CPU | 3000x5000 | 5000x4000 | 0.73 |
| Accelerate BLAS | 3000x5000 | 5000x4000 | 2011.5 |
| | | | |
| `mat_mul_simple1` | 20000x20000 | 20000x20000 | 555.8 |
| Naive CPU | 20000x20000 | 20000x20000 | x |
| Accelerate BLAS | 20000x20000 | 20000x20000 | 2090.8 |


Observations:
- The naive GPU version seems surprisingly fast considering that this is the most basic implementation with no optimizations. Once the matrix size is sufficiently large, it consitently runs at around 550+ GFLOPS. The performance is not good with smaller matrices such as 256x256, though, suggesting there could be some sort of overhead in calling the shader that starts to become the performance bottleneck if the shader does not perform enough work.
- Observe that the CPU results are quite poor here, with the GPU implementation running hundreds of times faster. It actually turns out that with a just little more work on the CPU side it is possible to make it run at approx 100 GFLOPs by simply computing a transposed multiplication to optimize memory access, which will enable automatic vectorization, and also using OpenMP to parallelize the outer loop. Since this would require either [experimental modifications to Apple's clang environment that could break at any time](https://mac.r-project.org/openmp/) or using a different compiler (e.g., clang on Homebrew), I did not include that code here. To be fair, we would then need to also modifiy the GPU shader to compute the transposed multiplication as well, which then increases the GPU performance to over 700 GFLOPs. So depending on whether we compare the naive or slightly optimized versions, the GPU speedup ranges from 5x - 1000x faster than the corresponding CPU code.
- The Accelerate BLAS sgemm version is still much faster than the naive GPU version, reaching around 2 TFLOPS for sufficiently large matrix sizes. Note that this actually uses the AMX accelerator, which I understand is external to the CPU but still contained on the M1 Max SOC.
- Calling Apple's `MPSMatrixMultiplication` kernel from Swift code results in roughly 7 TFLOPS of performance on the GPU for sufficiently large matrix sizes, showing that our naive GPU implementation here is still far below the potential acheivable performance. Apparantly, the peak FP32 performance of the M1 MAX GPU is slightly over 10 TFLOPS.
- I observed the best performance using a small 8x8 threadgroup size but did not experiment with it much.

--------------

#### Experiment 2: Performance of the optimized shader

Now let's see if we can optimize the GPU implementation to improve the performance. I think it is interesting to start with a working CUDA kernel for the following reasons:
- It can serve as a concrete example of porting a CUDA kernel to Metal. It will be interesting to see how much work is required to do the porting.
- We can then run both implementations on roughly similar hardware and check whether a kernel that performs well on CUDA will also perform well on Metal. Do any changes need to be made to get good performance on Metal?
- It can take a lot of work to develope and debug such optimized kernels, so I prefer to use existing working code if possible.

Fortunately, there seem to be several optimized example CUDA kernels in various matrix multiplication tutorials. Let's use  [this one](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/matrixMul/matrixMul.cu) from NVIDIA's cuda-samples, since you may have already seen it when first learning CUDA. It uses uses shared threadgroup memory and a tiled algorithm. 

I am happy to report that it was extremely straightforward to port to Metal. As you can see by comparing the corresponding Metal and CUDA kernel sources in `mat_mul_optimized_nv.metal`, they look very similar. The following are the key differences between them:
- Change `__syncthreads();` in CUDA to `threadgroup_barrier(mem_flags::mem_none);` in Metal.
- Change `__shared__` in CUDA to `threadgroup` in CUDA.
- In CUDA, you can simply call the kernel with arbitrary parameters. However, in Metal you can only pass in Buffer pointers to (shared) GPU memory which can then be cast to anything you like (as long as it is the correct thing). Don't worry, you will of course be careful to double check to avoid making an easy error here. To pass arbitrary parameters in Metal, the convention seems to be to make a custom struct containing them and supply it to the shader as a Buffer.
- The CUDA kernel is restricted to matrix sizes that are a multiple of the block size. I added a bounds check to the Metal version to support arbitrary sized matrices.

Now let's look at the performance results. This corresponds to the `shader_name = "mat_mul_optimized_nv"` section in `run_mat_mult_shaders()`. In addition to running the Metal code on the M1 Max, I ran the corresponding CUDA sample on an NVIDIA 1080 Ti, which seems to have roughly similar hardware performance in terms of GPU memory bandwidth and peak FP32 FLOPS.

First the Metal kernel: (threadgroup size is 8x8 unless otherwise specified)

| Implementation | A size | B size | GFLOPS |
|----------------|--------|--------|--------|
| `mat_mul_simple1` | 256x256 | 256x256 | 76.4 |
| `mat_mul_optimized_nv` | 256x256 | 256x256 |114.6 |
| CUDA 1080Ti | 256x256 | 256x256 |1034.1 |
| Accelerate BLAS | 256x256 | 256x256 | 1097.9 |
| | | | |
| `mat_mul_simple1` | 1230x789 | 789x1156 | 583.8 |
| `mat_mul_optimized_nv` | 1230x789 | 789x1156 | 889.3 |
| Accelerate BLAS | 1230x789 | 789x1156 | 1484.8 |
| | | | |
| `mat_mul_simple1` | 1024x1024 | 1024x1024 | 578.3 |
| `mat_mul_optimized_nv` | 1024x1024 | 1024x1024 | 1227.3 |
| CUDA 1080Ti | 1024x1024 | 1024x1024 | 1877.4 |
| Accelerate BLAS | 1024x1024 | 1024x1024 | 1724.8 |
| | | | |
| `mat_mul_simple1` | 2048x2048 | 2048x2048 | 652.5 |
| `mat_mul_optimized_nv` | 2048x2048 | 2048x2048 | 1390.4 |
| `mat_mul_optimized_nv` | 2048x2048 | 2048x2048 | 947.0 (using 32x32 threadgroup size) |
| CUDA 1080Ti | 2048x2048 | 2048x2048 | 1849.5 |
| Accelerate BLAS | 2048x2048 | 2048x2048 | 1883.4 |
| | | | |
| `mat_mul_simple1` | 3000x5000 | 5000x4000 | 553.8 |
| `mat_mul_optimized_nv` | 3000x5000 | 5000x4000 | 530.9 |
| Accelerate BLAS | 3000x5000 | 5000x4000 | 2011.5 |
| | | | |
| `mat_mul_simple1` | 20000x20000 | 20000x20000 | 555.8 |
| `mat_mul_optimized_nv` | 20000x20000 | 20000x20000 | 848.7  (using 32x32 threadgroup size) |
| `mat_mul_optimized_nv` | 20000x20000 | 20000x20000 | 122.0 |
| Accelerate BLAS | 20000x20000 | 20000x20000 | 2090.8 |

Observations:
- For appropriate threadgroup sizes, the optimized shader is significantly faster than the naive one.
- The optimized shader is still slower than the Accelerate BLAS.
- The CUDA kernel running on a 1080Ti still performed well at smaller matrix sizes such as 256x256, while the Metal shader slowed down significnatly.

--------------

#### Experiment 3: Interleaving CPU and GPU computations

Since the M1-based SOCs contain shared CPU-GPU memory, we should expect that there will be no performance penalty when the CPU and GPU take turns operating on the (shared) data. This is because unlike traditional discrete GPUs with physically separate memory from the CPU, there is no longer a need to maintain separate CPU and GPU arrays and copy data between them. Let's test this out by running the GPU matrix multiplicaiton operation inside a loop in which the GPU and CPU code take turns reading and/or writing to the same backing arrays each iteration. In a traditional GPU, this would require to copy the matrix values to/from the CPU and GPU each iteration, potentially slowing things down. If there is actually no copying of data between the CPU and GPU on the M1 Max, we should expect that performing these interleaved CPU and GPU operations should require the same amount of time that it would take to perform only the CPU subset of the operations plus the time that it would take to perform only the GPU subset of the operations. We will test this by doing just that and compute the timings.


The function `run_interleaved()` contains the code to run the experiments in this section. It uses only use the optimized `mat_mul_optimized_nv` shader for the GPU operation in these experiments. The CPU operations are described below.

Let's use the following arbitrary matrix sizes:
A: 1000x900
B: 900x1200

As a first step, we will verify that interleaving the GPU matrix multiplication with CPU code that only updates a few elements in each matrix has essentially no performance impact. We can do this by inserting a call to method `multiplier.touch_data_cpu()` which uses CPU code to modify a single element in each source matrix A and B. Running the code, we observe that adding the "touch" call has no effect on the timings.

Now let's change the touch operation to a more expensive CPU operation that operates on the whole backing arrays of the matrices. We will use the simple ReLU function (implemented in `compute_forward_relu_in_place()` of `Utilities.cpp`). It simply sets negative values in the Matrix to 0.

Here are the timings:

| GPU Multiply only | CPU ReLu only | Both interleaved |
|---|--|---|
| 413 millisec | 157 millisec | 568 millisec |

As expected, the sum of the column 1 (GPU operation only) + column 2 (CPU operation only) is approximately column 3 (interleaving both operations). At least for this example, it shows that there is no measurable overhead in interleaving operations on the CPU and GPU that need to access the same data in memory.

This was just a preliminary test, but so far the Apple Silicon Macs are looking promising for simplifying the memory management aspect of GPU programing and potentially improving performance by eliminating the CPU-GPU data transfers that were required with traditional discrete GPUs.



--------------
### Misc. Notes

- If you want to create an array to share between the CPU and GPU, it seems that it needs to be created by calling `newBuffer(buffer_size_in_bytes, MTL::ResourceStorageModeShared)` on the GPU's device pointer. This will allocate the storage and create a GPU pointer to the array (referred to as "buffer" in the code). You can then get the CPU pointer to the same array by calling `contents()` on it. That is, you first need to allocate the storage on the GPU side and then get a CPU pointer to it. There is apparantly only a single copy of the array in the shared physical memory and we now have two different types of pointers to it. Either the CPU or GPU should then be able to access and/or write data using their respective pointers at any time. Since the memory is physically shared, these reads and writes should not trigger any copying of the array between CPU and GPU (because there should be only a single copy in the shared memory).
- If anyone is aware of a faster optimized open source shader than the "optimized" one in this repo, I would be interested to know about it. Since Apple's `MPSMatrixMultiplication` kernel runs several times faster (I see around 7 TFLOPS when calling it from Swift code), I would think it should certainly be possible, assuming they used the public API to implement their proprietary kernel. If anyone even knows how to call `MPSMatrixMultiplication` from C++ code, I would also be interested to know.


### License

FreeBSD license.
