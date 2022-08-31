//
//  main.cpp
//  metal_performance_testing
//
//  Created by Brian Vogel on 2022/08/27.
//

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <iostream>
#include <chrono>

#include "MatrixMultiplier.h"

using namespace std;


void run_mat_mult_shaders() {
    // Perform matrix multiplication using custom shaders:
    // - simple custom shader with no optimization.
    // - optimized custom shader ported from existing CUDA kernel.
    //
    // Computes:
    // X <- A * B
    //
    // where the matrices are of floats and have the followqing sizes:
    //
    // X size: rows_X x cols_X
    // A size: rows_X x inner_dim
    // B size: inner_dim x cols_X
    
    // Arbitrary matrix sizes are supported!
    
    //const int rows_X = 256;
    //const int cols_X = 256;
    //const int inner_dim = 256;
    
    //const int rows_X = 1230;
    //const int cols_X = 1156;
    //const int inner_dim = 789;
    
    const int rows_X = 1024;
    const int cols_X = 1024;
    const int inner_dim = 1024;
    
    //const int rows_X = 2048;
    //const int cols_X = 2048;
    //const int inner_dim = 2048;
    
    //const int rows_X = 3000;
    //const int cols_X = 4000;
    //const int inner_dim = 5000;
    
    
    //const int rows_X = 20000;
    //const int cols_X = 20000;
    //const int inner_dim = 20000;
    
    
    //const int rows_X = 8192;
    //const int cols_X = 8192;
    //const int inner_dim = 8192;
    
    cout << "Running Experiments 1 and 2: matrix multiplication example with naive and optimized shaders." << endl;
    
    // Get the GPU device.
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
   
    // The name of the shader to run.
    const string shader_name = "mat_mul_simple1";
    
    MatrixMultiplier multiplier(device, shader_name);
    multiplier.allocate_memory(rows_X, cols_X, inner_dim);
    multiplier.initialize_data();
    
    // Perform the multiplication
    multiplier.run_multiply_on_gpu();
    
    // Verify that it computes the correct result
    multiplier.check_results();
    
    // Benchmark the Metal code
    
    {
        cout << "Running benchmark for Metal shader: " << shader_name << endl;
        using namespace std::chrono;
        auto t0 = high_resolution_clock::now();
        int loop_count = 200;
        for (int n = 0; n != loop_count; ++n)
        {
            // Perform the multiplication
            multiplier.run_multiply_on_gpu();
        }
        auto t1 = high_resolution_clock::now();
        auto time_in_usec = duration_cast<microseconds>(t1 - t0).count();
        double gflops = 2e-3 * static_cast<double>(loop_count)  * static_cast<double>(rows_X) * static_cast<double>(cols_X) * static_cast<double>(inner_dim) / static_cast<double>(time_in_usec);
        cout << gflops << " GFLOPS" << endl;
        cout << "\n-------------------------\n" << endl;
    }
    
    // Switch to the optimized shader.
    const string shader_name_optimized = "mat_mul_optimized_nv";
    multiplier.change_shader(shader_name_optimized);
    multiplier.initialize_data();
    // Perform the multiplication
    multiplier.run_multiply_on_gpu();
    // Verify that it computes the correct result
    multiplier.check_results();
    
    {
        cout << "Running benchmark for Metal shader: " << shader_name_optimized << endl;
        using namespace std::chrono;
        auto t0 = high_resolution_clock::now();
        int loop_count = 200;
        for (int n = 0; n != loop_count; ++n)
        {
            // Perform the multiplication
            multiplier.run_multiply_on_gpu();
        }
        auto t1 = high_resolution_clock::now();
        auto time_in_usec = duration_cast<microseconds>(t1 - t0).count();
        double gflops = 2e-3 * static_cast<double>(loop_count)  * static_cast<double>(rows_X) * static_cast<double>(cols_X) * static_cast<double>(inner_dim) / static_cast<double>(time_in_usec);
        cout << gflops << " GFLOPS" << endl;
        cout << "\n-------------------------\n" << endl;
    }
    
    const int naive_loop_count = 0; // Set to 0 for large matrix sizes because slow.
    if (naive_loop_count > 0) {
        cout << "Running benchmark for naive CPU"<< endl;
        using namespace std::chrono;
        auto t0 = high_resolution_clock::now();
        for (int n = 0; n != naive_loop_count; ++n)
        {
            // Perform the multiplication
            multiplier.run_on_cpu_naive_single_thread();
        }
        auto t1 = high_resolution_clock::now();
        auto time_in_usec = duration_cast<microseconds>(t1 - t0).count();
        double gflops = 2e-3 * static_cast<double>(naive_loop_count)  * static_cast<double>(rows_X) * static_cast<double>(cols_X) * static_cast<double>(inner_dim) / static_cast<double>(time_in_usec);
        cout << gflops << " GFLOPS" << endl;
        cout << "\n-------------------------\n" << endl;
    }
    
    
    {
        cout << "Running benchmark for Accelerate BLAS sgemm on CPU"<< endl;
        using namespace std::chrono;
        auto t0 = high_resolution_clock::now();
        int loop_count = 200;
        for (int n = 0; n != loop_count; ++n)
        {
            // Perform the multiplication
            multiplier.run_on_cpu_accelerate_blas();
        }
        auto t1 = high_resolution_clock::now();
        auto time_in_usec = duration_cast<microseconds>(t1 - t0).count();
        double gflops = 2e-3 * static_cast<double>(loop_count)  * static_cast<double>(rows_X) * static_cast<double>(cols_X) * static_cast<double>(inner_dim) / static_cast<double>(time_in_usec);
        cout << gflops << " GFLOPS" << endl;
    }
}

void run_interleaved() {
    // Run interleaved CPU/GPU exeriment (see README for description)
    //
    // Computes:
    // X <- A * B
    //
    // where the matrices are of floats and have the followqing sizes:
    //
    // X size: rows_X x cols_X
    // A size: rows_X x inner_dim
    // B size: inner_dim x cols_X
    //
    // and interleave it with "touch" or relu operation and compute timings.
    
    // Arbitrary matrix sizes are supported!
    
    //const int rows_X = 256;
    //const int cols_X = 256;
    //const int inner_dim = 256;
    
    //const int rows_X = 1230;
    //const int cols_X = 1156;
    //const int inner_dim = 789;
    
    //const int rows_X = 512;
    //const int cols_X = 512;
    //const int inner_dim = 512;
    
    const int rows_X = 1000;
    const int cols_X = 1200;
    const int inner_dim = 900;
    
    //const int rows_X = 2048;
    //const int cols_X = 2048;
    //const int inner_dim = 2048;
    
    //const int rows_X = 3000;
    //const int cols_X = 4000;
    //const int inner_dim = 5000;
    
    
    //const int rows_X = 20000;
    //const int cols_X = 20000;
    //const int inner_dim = 20000;
    
    
    //const int rows_X = 8192;
    //const int cols_X = 8192;
    //const int inner_dim = 8192;
    
    cout << "Running Experiment 3: Interleaving CPU and GPU computations." << endl;
    
    // Get the GPU device.
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
   
    // The name of the shader to run.
    const string shader_name = "mat_mul_optimized_nv";
    
    MatrixMultiplier multiplier(device, shader_name);
    multiplier.allocate_memory(rows_X, cols_X, inner_dim);
    multiplier.initialize_data();
    
    // Perform the multiplication
    multiplier.run_multiply_on_gpu();
    
    // Verify that it computes the correct result
    multiplier.check_results();
    
    // Benchmark the Metal code
    
    {
        cout << "Running multiplication only for Metal shader: " << shader_name << endl;
        using namespace std::chrono;
        auto t0 = high_resolution_clock::now();
        int loop_count = 200;
        for (int n = 0; n != loop_count; ++n)
        {
            multiplier.run_multiply_on_gpu();
        }
        auto t1 = high_resolution_clock::now();
        auto time_in_usec = duration_cast<microseconds>(t1 - t0).count();
        double gflops = 2e-3 * static_cast<double>(loop_count)  * static_cast<double>(rows_X) * static_cast<double>(cols_X) * static_cast<double>(inner_dim) / static_cast<double>(time_in_usec);
        cout << gflops << " GFLOPS" << endl;
        cout << "GPU Multiplication: " << time_in_usec*1e-3 << " milliseconds" << endl;
        cout << "\n-------------------------\n" << endl;
    }
    

    
    {
        cout << "Running multipliation and touch data for Metal shader: " << shader_name <<  endl;
        using namespace std::chrono;
        auto t0 = high_resolution_clock::now();
        int loop_count = 200;
        for (int n = 0; n != loop_count; ++n)
        {
            // Modify a tiny part of the matrix data.
            multiplier.touch_data_cpu();
            // Perform the multiplication
            multiplier.run_multiply_on_gpu();
        }
        auto t1 = high_resolution_clock::now();
        auto time_in_usec = duration_cast<microseconds>(t1 - t0).count();
        double gflops = 2e-3 * static_cast<double>(loop_count)  * static_cast<double>(rows_X) * static_cast<double>(cols_X) * static_cast<double>(inner_dim) / static_cast<double>(time_in_usec);
        cout << gflops << " GFLOPS" << endl;
        cout << "CPU touch + Multiplication: " << time_in_usec*1e-3 << " milliseconds" << endl;
        cout << "\n-------------------------\n" << endl;
    }
    
    {
        cout << "Running multipliation and relu data for Metal shader: " << shader_name <<  endl;
        using namespace std::chrono;
        auto t0 = high_resolution_clock::now();
        int loop_count = 200;
        for (int n = 0; n != loop_count; ++n)
        {
            // Modify a tiny part of the matrix data.
            multiplier.relu_cpu();
            // Perform the multiplication
            multiplier.run_multiply_on_gpu();
        }
        auto t1 = high_resolution_clock::now();
        auto time_in_usec = duration_cast<microseconds>(t1 - t0).count();
        double gflops = 2e-3 * static_cast<double>(loop_count)  * static_cast<double>(rows_X) * static_cast<double>(cols_X) * static_cast<double>(inner_dim) / static_cast<double>(time_in_usec);
        cout << gflops << " GFLOPS" << endl;
        cout << "Relu (CPU) + multiplication: " << time_in_usec*1e-3 << " milliseconds" << endl;
        cout << "\n-------------------------\n" << endl;
    }
    
    {
        cout << "Running CPU relu" <<  endl;
        using namespace std::chrono;
        auto t0 = high_resolution_clock::now();
        int loop_count = 200;
        for (int n = 0; n != loop_count; ++n)
        {
            // Modify a tiny part of the matrix data.
            multiplier.relu_cpu();
            
        }
        auto t1 = high_resolution_clock::now();
        auto time_in_usec = duration_cast<microseconds>(t1 - t0).count();
        //double gflops = 2e-3 * static_cast<double>(loop_count)  * static_cast<double>(rows_X) * static_cast<double>(cols_X) * static_cast<double>(inner_dim) / static_cast<double>(time_in_usec);
        //cout << gflops << " GFLOPS" << endl;
        cout << "Relu only (CPU): " << time_in_usec*1e-3 << " milliseconds" << endl;
        cout << "\n-------------------------\n" << endl;
    }
    
}

int main(int argc, char *argv[])
{
    // Matrix multiplication example using each shader.
    run_mat_mult_shaders();
    
    // Run interleaved computations on CPU and GPU.
    run_interleaved();
}
