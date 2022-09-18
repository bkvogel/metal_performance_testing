#ifndef _UTILITIES_H
#define _UTILITIES_H

#include "Matrix.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include "Assertions.h"

//
// Conventions:
//
// Input matrices to functions are specified as const references. Output matrices are specified as (non-const) references.
//
// Automatic resizing of output matrices:
//
// Most (eventually all) functions that modify a matrix will resize the matrix to the appropriate dimensions, if necessary.
// This feature makes these functions easier to use because the user is releived from the burden of having to
// determine and set the appropriate dimensions before the function call. For example, to compute the matrix product:
//
// A <- B x C
//
// The user can simply allocate an empty Matrix A such as
//
// Matrix A;
// mat_multiply(A, B, C); // Assume B and C are already available and filled with data.
//
// The matrix A will be empty (size 0) when passed in to the function, but will be resized to the appropriate
// dimensions during the function call.
//
// Other functions that take a modifiable matrix reference generally behave similarly.
//
// Since resizing a matrix can be expensive, it is good practice to write code such that resizing is only performed during
// the "initialization" stage. For debugging purposes, these functions can be configured to print a "resized" message to
// stdout whenever a matrix is resized by defining KUMOZU_DEBUG in the makefile.

////////////////////////////////////////////////////////////////////////////////////////////
// Matrix Utilities

/**
 * Compute B x C and place the result in A.
 *
 * Note: This method implements the basic easy-to-understand version. It is
 * not optimized in any way.
 *
 * Parameters
 *
 * @param A The result is returned in this matrix, which will be resized to the appropriate dimensions
 * if necessary.
 * @param B Input matrix which is not modified.
 * @param C Input matrix which is not modified.
 */
void mat_multiply_naive(MatrixF &A, const MatrixF &B, const MatrixF &C);

/**
 * Compute B x C and place the result in A.
 *
 * This implementation will call an optimized BLAS if one is available.
 *
 * @param A The result is returned in this matrix, which will be resized to the appropriate dimensions
 * if necessary.
 * @param B Input matrix which is not modified.
 * @param C Input matrix which is not modified.
 */
void mat_multiply(MatrixF &A, const MatrixF &B, const MatrixF &C);

/**
 * Compute A = alpha*B*C + beta*A.
 *
 * This implementation will call an optimized BLAS if one is available.
 *
 * @param A The result is returned in this matrix, which will be resized to the appropriate dimensions
 * if necessary.
 * @param B Input matrix which is not modified.
 * @param C Input matrix which is not modified.
 */
void mat_multiply(MatrixF &A, const MatrixF &B, const MatrixF &C, float alpha, float beta);

/**
 * Compute A = B*C using an optimized BLAS.
 *
 * @param A The result is returned in this matrix, which will be resized to the appropriate dimensions
 * if necessary.
 * @param B Input matrix which is not modified.
 * @param C Input matrix which is not modified.
 */
void mat_multiply_blas(MatrixF &A, const MatrixF &B, const MatrixF &C);

/**
 * Compute A = alpha*B*C + beta*A using an optimzied BLAS.
 *
 * @param A The result is returned in this matrix, which will be resized to the appropriate dimensions
 * if necessary.
 * @param B Input matrix which is not modified.
 * @param C Input matrix which is not modified.
 */
void mat_multiply_blas(MatrixF &A, const MatrixF &B, const MatrixF &C, float alpha, float beta);

/**
 * Set all values to be uniformly disstributed random values in [min, max].
 *
 */
void randomize_uniform(MatrixF &A, float min, float max);


void assert_almost_equal_relative_error(const MatrixF &A, const MatrixF &B, float tolerance = 1.0e-3f);


/*
 * For each element of the two suplied matrices, which must be of the same size,
 * test if the magnitude of the difference exceeds the tolerance. If so,
 * exit with an error.
 */
float assert_almost_equal_max_error(const MatrixF &A, const MatrixF &B, float tolerance = 1.0e-3f);

/*
 * Given a loop count and a function, call the function loop_count
 * times and print the mean time per call.
 *
 * Returns:
 #  The mean time per call in microseconds.
 */
template <typename Func>
float benchmark(int loop_count, Func func) {
    // Warmup:
    func();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int n = 0; n != loop_count; ++n) {
        func();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto time_in_microseconds = duration_cast<std::chrono::microseconds>(t1 - t0).count();
    float time_per_call = static_cast<float>(time_in_microseconds)/static_cast<float>(loop_count);
    std::cout << "Time per call: " << time_per_call << " microseconds" << std::endl;
    return time_per_call;
}

/**
 * Return the maximum value.
 */
template <typename T>
T max_value(const Matrix<T> &A)
{
    assertion(A.size() > 0, "max_value(): Cannot call on size-0 matrix.");
    T max_val = A[0];
    for (auto i = 0; i != A.size(); ++i)
    {
        max_val = std::max(max_val, A[i]);
    }
    return max_val;
}

/**
 * Return the minimum value.
 */
template <typename T>
T min_value(const Matrix<T> &A)
{
    assertion(A.size() > 0, "min_value(): Cannot call on size-0 matrix.");
    T min_val = A[0];
    for (auto i = 0; i != A.size(); ++i)
    {
        min_val = std::min(min_val, A[i]);
    }
    return min_val;
}

/**
 * Print all elements in the vector to std out.
 */
template <typename T>
void print_vector(std::vector<T> vec)
{
    for_each(begin(vec), end(vec), [](T val)
             { std::cout << val << " "; });
    std::cout << std::endl;
}

/**
 * Check that both matrices have the same dimensions. If they differ, exit with an error.
 */
template <typename T1, typename T2>
void check_dimensions(const Matrix<T1> &A, const Matrix<T2> &B)
{
    if (A.get_extents() != B.get_extents())
    {
        std::cerr << "A extents: " << std::endl;
        print_vector(A.get_extents());
        std::cerr << "B extents: " << std::endl;
        print_vector(B.get_extents());
        error_exit("Error: Supplied matrices A and B do not have the same extents!");
    }
}



/**
 * Use method from: http://cs231n.github.io/neural-networks-3/
 *
 * |A - B|
 * ---------
 * max(|A|, |B|)
 *
 * Note: do not use this method if we expect both matrices to be nearly 0.
 */
float relative_error(const MatrixF &A, const MatrixF &B);

/**
 * Set all elements of the matrix <i>A</i> to have value <i>value</i> and
 * return the result in <i>A</i>.
 * @param A Input and output matrix.
 * @param value The value to set.
 */
template <typename T1, typename T2>
void set_value(Matrix<T1>& A, T2 value) {
//#pragma omp parallel for
    for (int i = 0; i < A.size(); i++) {
        A[i] = value;
    }
}

/**
 * Forward-direction ReLU (Rectified Linear Unit) activation function.
 *
 *
 * The function computed is x[i] = max(0, x[i]) for all indices i in the backing array.
 *
 *
 * Parameters:
 *
 * @param x The input and output matrix.
 *
 *
 */
 void compute_forward_relu_in_place(MatrixF &x);

#endif /* _UTILITIES_H */
