//
//  mat_mul_optimized_nv.metal
//  metal_performance_testing
//
//  Created by Brian Vogel on 2022/08/27.
//

#include <metal_stdlib>
#include "ShaderParams.h"
using namespace metal;
 
/**
 * Compute matrix multiplication result = inA x inB.
 * This is an optimized matrix multiplication shader using shared threadgroup memory with a
 * tiling algorithm. It was ported from the CUDA kernel at the end of this file.
 *
 * Requirements:
 * - All matrices are assumed to have row-major ordering.
 * - All matrix dimensions must be an integer multiple of the block size.
 *
 */
kernel void mat_mul_optimized_nv(device const float* inA,
                                 device const float* inB,
                                 device float* result,
                                 constant MatMulParams& params,
                                 uint2 threadgroup_pos [[ threadgroup_position_in_grid ]],
                                 uint2 local_thread_idx [[ thread_position_in_threadgroup ]],
                                 uint2 id [[ thread_position_in_grid ]])
{
    
    // Note: be sure that this is set to the same value as "threads per group" in the calling code!
    const int BLOCK_SIZE = 8;

    const uint wB = params.col_dim_x;
    const uint wA = params.inner_dim;
    
    // Block index
    const uint bx =threadgroup_pos.x;
    const uint by = threadgroup_pos.y;
    
    // Thread index
    const uint tx =local_thread_idx.x;
    const uint ty =local_thread_idx.y;
    
    // Index of the first sub-matrix of A processed by the block
    const uint aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    const uint aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    const uint aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    const uint bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    const uint bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;
    
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (uint a = aBegin, b = bBegin;
        a <= aEnd;
        a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        threadgroup float As[BLOCK_SIZE][BLOCK_SIZE];
        

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        threadgroup float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = inA[a + wA * ty + tx];
        Bs[ty][tx] = inB[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        threadgroup_barrier(mem_flags::mem_none);

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        threadgroup_barrier(mem_flags::mem_none);
      }

    const int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    result[c + wB * ty + tx] = Csub;    
}

// For reference, I have copied the original CUDA kernel source code below, which
// is included in the "cuda-samples" package: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/matrixMul/matrixMul.cu
// Key differences in syntax:
// - Change `__syncthreads();` in CUDA to `threadgroup_barrier(mem_flags::mem_none);` in Metal.
// - Change `__shared__` in CUDA to `threadgroup` in CUDA.
// - In CUDA, you can simply call the kernel with arbitrary parameters. However, in Metal you
// can only pass in Buffer pointers to (shared) GPU memory which can then be cast to anything you like.
// To pass arbitrary parameters in Metal, the convention seems to be to make a custom struct containing them
// and supply it to the shader as a Buffer.

// Original CUDA kernel source from NVIDIA (Note that the license allows redistribution):

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling approach.
 * It has been written for clarity of exposition to illustrate various CUDA programming
 * principles, not with the goal of providing the most performant generic kernel for matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */
/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
/*
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A,
    float *B, int wA,
    int wB) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Index of the first sub-matrix of A processed by the block
  int aBegin = wA * BLOCK_SIZE * by;

  // Index of the last sub-matrix of A processed by the block
  int aEnd   = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep  = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE * bx;

  // Step size used to iterate through the sub-matrices of B
  int bStep  = BLOCK_SIZE * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin;
       a <= aEnd;
       a += aStep, b += bStep) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll

    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + wB * ty + tx] = Csub;
}
*/
