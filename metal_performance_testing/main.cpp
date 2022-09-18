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

#include "Utilities.h"
#include "MatrixMultiplier.h"

using namespace std;


float matmul_time_to_gflops(float rows, float cols, float inner_dim, float microsecs) {
    return 2e-3  * static_cast<float>(rows) * static_cast<float>(cols) * static_cast<float>(inner_dim) / static_cast<float>(microsecs);
}

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
    
    //const int rows_X = 1024;
    //const int cols_X = 768;
    //const int inner_dim = 512;
    
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
    
    cout << "Running Experiments 1 through 3: matrix multiplication example with naive and optimized shaders." << endl;
    
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
    
    const int loop_count = 200;
    
    float microsec_per_call;
    // Benchmark the Metal code
    
    cout << "Running benchmark for Metal shader: " << shader_name << endl;
    microsec_per_call = benchmark(loop_count, [&] () {
        // Perform the multiplication
        multiplier.run_multiply_on_gpu();
    });
    cout << matmul_time_to_gflops(rows_X, cols_X, inner_dim, microsec_per_call) << " GFLOPS" << endl;
    cout << "\n-------------------------\n" << endl;
    
    // Switch to the NVIDIA example optimized shader.
    const string shader_name_nv_optimized = "mat_mul_optimized_nv";
    multiplier.change_shader(shader_name_nv_optimized);
    multiplier.initialize_data();
    // Perform the multiplication
    multiplier.run_multiply_on_gpu();
    // Verify that it computes the correct result
    multiplier.check_results();
    
    cout << "Running benchmark for Metal shader: " << shader_name_nv_optimized << endl;
    microsec_per_call = benchmark(loop_count, [&] () {
        // Perform the multiplication
        multiplier.run_multiply_on_gpu();
    });
    cout << matmul_time_to_gflops(rows_X, cols_X, inner_dim, microsec_per_call) << " GFLOPS" << endl;
    cout << "\n-------------------------\n" << endl;
    
    // Switch to the my optimized shader v1.
    const string shader_name_mat_mul_opt1 = "mat_mul_opt1";
    multiplier.change_shader(shader_name_mat_mul_opt1);
    multiplier.initialize_data();
    // Perform the multiplication
    multiplier.run_multiply_on_gpu_mat_mul_opt1();
    // Verify that it computes the correct result
    multiplier.check_results();
    
    cout << "Running benchmark for Metal shader: " << shader_name_mat_mul_opt1 << endl;
    microsec_per_call = benchmark(loop_count, [&] () {
        // Perform the multiplication
        multiplier.run_multiply_on_gpu_mat_mul_opt1();
    });
    cout << matmul_time_to_gflops(rows_X, cols_X, inner_dim, microsec_per_call) << " GFLOPS" << endl;
    cout << "\n-------------------------\n" << endl;
    
    // Switch to the my optimized shader v2.
    const string shader_name_mat_mul_opt2 = "mat_mul_opt2";
    multiplier.change_shader(shader_name_mat_mul_opt2);
    multiplier.initialize_data();
    // Perform the multiplication
    multiplier.run_multiply_on_gpu_mat_mul_opt2();
    // Verify that it computes the correct result
    multiplier.check_results();
    
    cout << "Running benchmark for Metal shader: " << shader_name_mat_mul_opt2 << endl;
    microsec_per_call = benchmark(loop_count, [&] () {
        // Perform the multiplication
        multiplier.run_multiply_on_gpu_mat_mul_opt2();
    });
    cout << matmul_time_to_gflops(rows_X, cols_X, inner_dim, microsec_per_call) << " GFLOPS" << endl;
    cout << "\n-------------------------\n" << endl;
    
    const int naive_loop_count = 0; // Set to 0 for large matrix sizes because slow.
    if (naive_loop_count > 0) {
        cout << "Running benchmark for naive CPU"<< endl;
        microsec_per_call = benchmark(loop_count, [&] () {
            // Perform the multiplication
            multiplier.run_on_cpu_naive_single_thread();
        });
        cout << matmul_time_to_gflops(rows_X, cols_X, inner_dim, microsec_per_call) << " GFLOPS" << endl;
        cout << "\n-------------------------\n" << endl;
    }
    
    cout << "Running benchmark for Accelerate BLAS sgemm on CPU"<< endl;
    microsec_per_call = benchmark(loop_count, [&] () {
        // Perform the multiplication
        multiplier.run_on_cpu_accelerate_blas();
    });
    cout << matmul_time_to_gflops(rows_X, cols_X, inner_dim, microsec_per_call) << " GFLOPS" << endl;
    cout << "\n-------------------------\n" << endl;
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
    
    cout << "Running Experiment 5: Interleaving CPU and GPU computations." << endl;
    
    // Get the GPU device.
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    
    // The name of the shader to run.
    const string shader_name = "mat_mul_opt2";
    
    MatrixMultiplier multiplier(device, shader_name);
    multiplier.allocate_memory(rows_X, cols_X, inner_dim);
    multiplier.initialize_data();
    
    // Perform the multiplication
    if (shader_name == "mat_mul_opt1") {
        multiplier.run_multiply_on_gpu_mat_mul_opt1();
    } if (shader_name == "mat_mul_opt2") {
        multiplier.run_multiply_on_gpu_mat_mul_opt2();
    }
    
    // Verify that it computes the correct result
    multiplier.check_results();
    
    // Benchmark the Metal code
    cout << "Running multiplication only for Metal shader: " << shader_name << endl;
    int loop_count = 200;
    float microsec_per_call;
    microsec_per_call = benchmark(loop_count, [&] () {
        // Perform the multiplication
        if (shader_name == "mat_mul_opt1") {
            multiplier.run_multiply_on_gpu_mat_mul_opt1();
        } if (shader_name == "mat_mul_opt2") {
            multiplier.run_multiply_on_gpu_mat_mul_opt2();
        }
    });
    cout << matmul_time_to_gflops(rows_X, cols_X, inner_dim, microsec_per_call) << " GFLOPS" << endl;
    cout << "\n-------------------------\n" << endl;
    
    cout << "Running multipliation and touch data for Metal shader: " << shader_name <<  endl;
    microsec_per_call = benchmark(loop_count, [&] () {
        // Modify a tiny part of the matrix data.
        multiplier.touch_data_cpu();
        // Perform the multiplication
        if (shader_name == "mat_mul_opt1") {
            multiplier.run_multiply_on_gpu_mat_mul_opt1();
        } if (shader_name == "mat_mul_opt2") {
            multiplier.run_multiply_on_gpu_mat_mul_opt2();
        }
    });
    cout << matmul_time_to_gflops(rows_X, cols_X, inner_dim, microsec_per_call) << " GFLOPS" << endl;
    cout << "CPU touch + Multiplication: " << microsec_per_call*1e-3 << " milliseconds" << endl;
    cout << "\n-------------------------\n" << endl;
    
    cout << "Running multipliation and relu data for Metal shader: " << shader_name <<  endl;
    microsec_per_call = benchmark(loop_count, [&] () {
        multiplier.relu_cpu();
        // Perform the multiplication
        if (shader_name == "mat_mul_opt1") {
            multiplier.run_multiply_on_gpu_mat_mul_opt1();
        } if (shader_name == "mat_mul_opt2") {
            multiplier.run_multiply_on_gpu_mat_mul_opt2();
        }
    });
    cout << matmul_time_to_gflops(rows_X, cols_X, inner_dim, microsec_per_call) << " GFLOPS" << endl;
    cout << "Relu (CPU) + multiplication: " << microsec_per_call*1e-3 << " milliseconds" << endl;
    cout << "\n-------------------------\n" << endl;
    
    cout << "Running CPU relu" <<  endl;
    microsec_per_call = benchmark(loop_count, [&] () {
        multiplier.relu_cpu();
    });
    cout << "Relu only (CPU): " << microsec_per_call*1e-3 << " milliseconds" << endl;
    cout << "\n-------------------------\n" << endl;
}

int main(int argc, char *argv[])
{
    // Matrix multiplication example using each shader.
    run_mat_mult_shaders();
    
    // Run interleaved computations on CPU and GPU.
    run_interleaved();
}
