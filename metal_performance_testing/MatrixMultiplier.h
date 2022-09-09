//
//  MatrixMultiplier.h
//  metal_performance_testing
//
//  Created by Brian Vogel on 2022/08/27.
//

#ifndef MatrixMultiplier_h
#define MatrixMultiplier_h

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"

class MatrixMultiplier
{
public:
    MatrixMultiplier(MTL::Device *device, std::string shader_name);
    
    /**
     * Switch to a different shader.
     */
    void change_shader(std::string shader_name);
    
    /**
     * Allocate the backing arrays for the matrices.
     *
     */
    void allocate_memory(int rows_X, int cols_X, int inner_dim);
    
    /**
     * Initialize the data arrays in the matrices.
     */
    void initialize_data();
    
    /**
     * Just modify a tiny part of the data on the CPU side.
     */
    void touch_data_cpu();
    
    
    void relu_cpu();
    
    /**
     * Run the Metal shader on GPU.
     */
    void run_multiply_on_gpu();
    
    /**
     * Run shader "mat_mul_opt1".
     */
    void run_multiply_on_gpu_mat_mul_opt1();
    
    /**
     * Run shader "mat_mul_opt2".
     */
    void run_multiply_on_gpu_mat_mul_opt2();
    
    /**
     * Run the multiplication on the CPU (actually AMX units) using
     *  cblas_sgemm() in Apple's Accelerate framework.
     */
    void run_on_cpu_accelerate_blas();
    
    /**
     * Run the multiplication on the CPU using the most basic implementation
     * without multithreading.
     */
    void run_on_cpu_naive_single_thread();
    
    void check_results();
    
private:
    MTL::CommandQueue *m_CommandQueue;
    MTL::Device *m_device_ptr;
    MTL::ComputePipelineState *m_MatMultiplyFunctionPSO;

    // GPU device buffers:
    // Device pointer to contents of matrix A.
    MTL::Buffer *m_device_buffer_A_ptr;
    // Device pointer to contents of matrix B.
    MTL::Buffer *m_device_buffer_B_ptr;
    // Device pointer to contents of matrix B.
    MTL::Buffer *m_device_buffer_X_ptr;
    // Device pointer to struct containing the shader parameters.
    MTL::Buffer* m_device_buffer_params_ptr;
    
    
    int m_rows_X;
    int m_cols_X;
    int m_cols_A;
};

#endif /* MatrixMultiplier_h */
