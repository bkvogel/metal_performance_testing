//
//  MatrixMultiplier.cpp
//  metal_performance_testing
//
//  Created by Brian Vogel on 2022/08/27.
//

#include "Matrix.h"
#include "MatrixMultiplier.h"
#include "ShaderParams.h"
#include "Utilities.h"
#include <iostream>
#include <stdio.h>

using namespace std;


MatrixMultiplier::MatrixMultiplier(MTL::Device *device, string shader_name) {
    m_device_ptr = device;
    change_shader(shader_name);
}

void MatrixMultiplier::change_shader(std::string shader_name) {
    MTL::Library *defaultLibrary = m_device_ptr->newDefaultLibrary();

    if (defaultLibrary == nullptr)
    {
        cout << "Failed to find the default library." << endl;
        return;
    }

    auto str = NS::String::string(shader_name.c_str(), NS::ASCIIStringEncoding);
    MTL::Function *matMultiplyFunction = defaultLibrary->newFunction(str);

    if (matMultiplyFunction == nullptr)
    {
        cout << "Failed to find the matrix multiplication shader." << endl;
        return;
    }

    NS::Error *error;
    m_MatMultiplyFunctionPSO = m_device_ptr->newComputePipelineState(matMultiplyFunction, &error);
    
    if (m_MatMultiplyFunctionPSO == nullptr)
    {
        cout << "Failed to create the PSO: " << error << endl;
        return;
    }
    
    NS::UInteger thread_execution_width = m_MatMultiplyFunctionPSO->threadExecutionWidth();
    cout << "FYI, the thread execution wdith is: " << thread_execution_width << endl;
    NS::UInteger max_total_threads_per_threadgroup = m_MatMultiplyFunctionPSO->maxTotalThreadsPerThreadgroup();
    cout << "FYI, the maximum allowed threads per threadgoup is: " << max_total_threads_per_threadgroup << endl;

    m_CommandQueue = m_device_ptr->newCommandQueue();
    if (m_CommandQueue == nullptr)
    {
        cout << "Failed to get the command queue." << endl;
        return;
    }
}

void MatrixMultiplier::allocate_memory(int rows_X, int cols_X, int inner_dim) {
    m_rows_X = rows_X;
    m_cols_X = cols_X;
    m_cols_A = inner_dim;
    
    // Allocate shared GPU/CPU buffers for the matrices.
    m_device_buffer_A_ptr = m_device_ptr->newBuffer(m_rows_X * m_cols_A * sizeof(float), MTL::ResourceStorageModeShared);
    m_device_buffer_B_ptr = m_device_ptr->newBuffer(m_cols_A * m_cols_X * sizeof(float), MTL::ResourceStorageModeShared);
    m_device_buffer_X_ptr = m_device_ptr->newBuffer(m_rows_X * m_cols_X * sizeof(float), MTL::ResourceStorageModeShared);
    
    // This is how we pass parameter values into our custom shader.
    // Steps to pass parameters (other than the data buffers) to a kernel:
    // - Decide which parameters the custom shader needs and make a struct to contain them.
    // - You can also define the struct in a header file and include it here and in the .metal files to avoid code duplication
    // - Allocate enough device buffer storage to contain the struct and reuse it.
    // Create a device buffer with enough storage for our MatMulParams struct.
    m_device_buffer_params_ptr = m_device_ptr->newBuffer(sizeof(MatMulParams), MTL::ResourceStorageModeShared);
}

void MatrixMultiplier::initialize_data()
{
    // Get CPU pointers to the same buffers and create a Matrix for each one.
    // Note that this apparantly does not make a "CPU copy" of the data but instead just
    // provides a pointer to the same underlying data that can be used by CPU code.
    
    // Create a Matrix of floats passing the "CPU" buffer pointer for
    // its backing array. The Matrix class uses row-major ordering.
    Matrix<float> A(static_cast<float*>(m_device_buffer_A_ptr->contents()), {m_rows_X, m_cols_A});
    Matrix<float> B(static_cast<float*>(m_device_buffer_B_ptr->contents()), {m_cols_A, m_cols_X});
    
    // Let's randomize the two input matricies.
    // This runs on the CPU (refer to the implementation in Utilities.cpp)
    randomize_uniform(A, -1.0f, 1.0f);
    randomize_uniform(B, -1.0f, 1.0f);
    
    // Or you could set all values to be the same for easier debugging:
    //set_value(A, 3.0);
    //set_value(B, 2.0);
    
    // The matricies are now initialized with random values.
    // Note that even though we initialized them
    // on the CPU and the next computations on them will happen on the GPU, we do not need to copy
    // the initialized values back to the GPU.
}

void MatrixMultiplier::touch_data_cpu() {
    Matrix<float> A(static_cast<float*>(m_device_buffer_A_ptr->contents()), {m_rows_X, m_cols_A});
    Matrix<float> B(static_cast<float*>(m_device_buffer_B_ptr->contents()), {m_cols_A, m_cols_X});
    Matrix<float> X(static_cast<float*>(m_device_buffer_X_ptr->contents()), {m_rows_X, m_cols_X}); // not used
    // Let's now just modify a singel value in each matrix, which requires almost no computation.
    A(0, 0) *= 0.9;
    B(0, 0) *= 0.95;
}

void MatrixMultiplier::relu_cpu() {
    Matrix<float> A(static_cast<float*>(m_device_buffer_A_ptr->contents()), {m_rows_X, m_cols_A});
    Matrix<float> B(static_cast<float*>(m_device_buffer_B_ptr->contents()), {m_cols_A, m_cols_X});
    Matrix<float> X(static_cast<float*>(m_device_buffer_X_ptr->contents()), {m_rows_X, m_cols_X}); // not used
    compute_forward_relu_in_place(A);
    compute_forward_relu_in_place(B);
}


void MatrixMultiplier::run_multiply_on_gpu()
{
    MatMulParams *params = (MatMulParams *)m_device_buffer_params_ptr->contents();
    params->row_dim_x = m_rows_X;
    params->col_dim_x = m_cols_X;
    params->inner_dim = m_cols_A;
    // Setup
    MTL::CommandBuffer *commandBuffer = m_CommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);

    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);

    computeEncoder->setComputePipelineState(m_MatMultiplyFunctionPSO);
    computeEncoder->setBuffer(m_device_buffer_A_ptr, 0, 0);
    computeEncoder->setBuffer(m_device_buffer_B_ptr, 0, 1);
    computeEncoder->setBuffer(m_device_buffer_X_ptr, 0, 2);
    computeEncoder->setBuffer(m_device_buffer_params_ptr, 0, 3);

    // Note: The kernel thread's 'x' position in the grid corresponds to the column index in the result matrix
    // and the 'y' position corresponds to the row index. Note that the matrix is in row-major so that the
    // column index is the "fast" index.
    
    // 8-32 threads per dim per group seem to work fine.
    // Both of these values must be the same!
    const int x_threads_per_group = 8;
    const int y_threads_per_group = 8;
    assert(x_threads_per_group == y_threads_per_group);
    
    // The number of thread groups (i.e., blocks) per grid.
    const int x_group_count = (m_cols_X + x_threads_per_group - 1) / x_threads_per_group;
    const int y_group_count = (m_rows_X + y_threads_per_group - 1) / y_threads_per_group;
    MTL::Size thread_group_count = MTL::Size::Make(x_group_count, y_group_count, 1); // should be the size of the grid = (x_threads, y_threads)
    MTL::Size threadgroupSize = MTL::Size::Make(x_threads_per_group, y_threads_per_group, 1); //
    computeEncoder->dispatchThreadgroups(thread_group_count, threadgroupSize);
    computeEncoder->endEncoding();

    // Start the shader!
    commandBuffer->commit();
    // Shader is still running here. Put other code here if you like.
    commandBuffer->waitUntilCompleted();
}

void MatrixMultiplier::run_multiply_on_gpu_mat_mul_opt1()
{
    MatMulParams *params = (MatMulParams *)m_device_buffer_params_ptr->contents();
    params->row_dim_x = m_rows_X;
    params->col_dim_x = m_cols_X;
    params->inner_dim = m_cols_A;
    // Setup
    MTL::CommandBuffer *commandBuffer = m_CommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);

    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);

    computeEncoder->setComputePipelineState(m_MatMultiplyFunctionPSO);
    computeEncoder->setBuffer(m_device_buffer_A_ptr, 0, 0);
    computeEncoder->setBuffer(m_device_buffer_B_ptr, 0, 1);
    computeEncoder->setBuffer(m_device_buffer_X_ptr, 0, 2);
    computeEncoder->setBuffer(m_device_buffer_params_ptr, 0, 3);

    // Note: The kernel thread's 'x' position in the grid corresponds to the column index in the result matrix
    // and the 'y' position corresponds to the row index.
    
    // 8-16 threads per dim per group seem to work well. These do not have to be the same value.
    const int x_threads_per_group = 8; // 16
    const int y_threads_per_group = 8; // 16
    
    const int y_submat_dim = 4; // don't change!
    const int num_row_threads = (m_rows_X + y_submat_dim - 1)/y_submat_dim;
    
    const int x_submat_dim = 4; // don't change!
    const int num_cols_threads = (m_cols_X + x_submat_dim - 1)/x_submat_dim;
    const int x_group_count = (num_cols_threads + x_threads_per_group - 1) / x_threads_per_group;
    const int y_group_count = (num_row_threads + y_threads_per_group - 1) / y_threads_per_group;
    
    MTL::Size thread_group_count = MTL::Size::Make(x_group_count, y_group_count, 1); // should be the size of the grid = (x_threads, y_threads)
    MTL::Size threadgroupSize = MTL::Size::Make(x_threads_per_group, y_threads_per_group, 1); //
    computeEncoder->dispatchThreadgroups(thread_group_count, threadgroupSize);
    computeEncoder->endEncoding();

    // Start the shader!
    commandBuffer->commit();
    // Shader is still running here. Put other code here if you like.
    commandBuffer->waitUntilCompleted();
}

void MatrixMultiplier::run_multiply_on_gpu_mat_mul_opt2()
{
    // Set the parameter values of the struct with the values you want to send to the shader (see below).
    MatMulParams *params = (MatMulParams *)m_device_buffer_params_ptr->contents();
    params->row_dim_x = m_rows_X;
    params->col_dim_x = m_cols_X;
    params->inner_dim = m_cols_A;
    // Setup
    MTL::CommandBuffer *commandBuffer = m_CommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);

    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);

    computeEncoder->setComputePipelineState(m_MatMultiplyFunctionPSO);
    computeEncoder->setBuffer(m_device_buffer_A_ptr, 0, 0);
    computeEncoder->setBuffer(m_device_buffer_B_ptr, 0, 1);
    computeEncoder->setBuffer(m_device_buffer_X_ptr, 0, 2);
    computeEncoder->setBuffer(m_device_buffer_params_ptr, 0, 3);

    // Note: The kernel thread's 'x' position in the grid corresponds to the column index in the result matrix
    // and the 'y' position corresponds to the row index. Note that the matrix is in row-major so that the
    // column index is the "fast" index.
    
    // 8-16 threads per dim per group seem to work well.
    const int x_threads_per_group = 16; // 16
    const int y_threads_per_group = 8; // 8
    
    const int y_submat_dim = 8;
    const int num_row_threads = (m_rows_X + y_submat_dim - 1)/y_submat_dim;
    
    const int x_submat_dim = 4;
    const int num_cols_threads = (m_cols_X + x_submat_dim - 1)/x_submat_dim;
    const int x_group_count = (num_cols_threads + x_threads_per_group - 1) / x_threads_per_group;
    const int y_group_count = (num_row_threads + y_threads_per_group - 1) / y_threads_per_group;
    
    MTL::Size thread_group_count = MTL::Size::Make(x_group_count, y_group_count, 1); // should be the size of the grid = (x_threads, y_threads)
    MTL::Size threadgroupSize = MTL::Size::Make(x_threads_per_group, y_threads_per_group, 1); //
    computeEncoder->dispatchThreadgroups(thread_group_count, threadgroupSize);
    computeEncoder->endEncoding();

    // Start the shader!
    commandBuffer->commit();
    // Shader is still running here. Put other code here if you like.
    commandBuffer->waitUntilCompleted();
}

void MatrixMultiplier::run_on_cpu_accelerate_blas() {
    // Run optimized CPU (AMX) BLAS.
    
    // Note that the following Matrix instances do not allocate new memory. They will use
    // the same shared backing array as the GPU. This is because they accept the supplied
    // CPU pointer to refer to their backing array. The following commands should therefore
    // not have any significnat overhead.
    Matrix<float> A(static_cast<float*>(m_device_buffer_A_ptr->contents()), {m_rows_X, m_cols_A});
    Matrix<float> B(static_cast<float*>(m_device_buffer_B_ptr->contents()), {m_cols_A, m_cols_X});
    Matrix<float> X(static_cast<float*>(m_device_buffer_X_ptr->contents()), {m_rows_X, m_cols_X});
    // Create empty matrix.
    Matrix<float> X_true;
    
    // Compute the matrix product using BLAS sgemm.
    // X_true <- A x B
    mat_multiply_blas(X_true, A, B);
}

void MatrixMultiplier::run_on_cpu_naive_single_thread() {
    // Run the most basic CPU implementation.
    
    // Note that the following Matrix instances do not allocate new memory. They will use
    // the same shared backing array as the GPU. This is because they accept the supplied
    // CPU pointer to refer to their backing array. The following commands should therefore
    // not have any significnat overhead.
    Matrix<float> A(static_cast<float*>(m_device_buffer_A_ptr->contents()), {m_rows_X, m_cols_A});
    Matrix<float> B(static_cast<float*>(m_device_buffer_B_ptr->contents()), {m_cols_A, m_cols_X});
    Matrix<float> X(static_cast<float*>(m_device_buffer_X_ptr->contents()), {m_rows_X, m_cols_X});
    // Create empty matrix.
    Matrix<float> X_true;
    
    // Compute the matrix product using naive CPU.
    // X_true <- A x B
    mat_multiply_naive(X_true, A, B);
}

void MatrixMultiplier::check_results()
{
    cout << "Verifying result..." << endl;
    Matrix<float> A(static_cast<float*>(m_device_buffer_A_ptr->contents()), {m_rows_X, m_cols_A});
    Matrix<float> B(static_cast<float*>(m_device_buffer_B_ptr->contents()), {m_cols_A, m_cols_X});
    Matrix<float> X(static_cast<float*>(m_device_buffer_X_ptr->contents()), {m_rows_X, m_cols_X});
    // Show the contents if small.
    if (X.size() < 1000) {
        cout << "A:\n" << A << endl;
        cout << "B:\n" << B << endl;
        cout << "X:\n" << X << endl;
    }
    
    const float max_allowable_error = 1e-3;
    
    // Create empty matrix.
    Matrix<float> X_true;
    
    // Compute the true matrix product using BLAS sgemm.
    // X_true <- A x B
    mat_multiply_blas(X_true, A, B);
    
    const float max_error = assert_almost_equal_max_error(X, X_true, max_allowable_error);
    
    const float max_result_val = max_value(X_true);
    if (max_result_val == 0) {
        cout << "Max result magnitude was: " << max_result_val << endl;
        cout << "It is meaningless to verify unless some values are non-zero!" << endl;
        error_exit("exiting");
    }
    cout << "Passed! Max error was: " << max_error << endl;
}
