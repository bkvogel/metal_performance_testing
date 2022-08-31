#include "Utilities.h"
#include <string>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <random>
#include <ctime>

// Use OpenBLAS for optimized matrix multiplication:
//#include <cblas.h>
// Use MKL for optimized matrix multiplication:
//#include "mkl.h"
// Use Accelerate for optimized matrix multiplication:
#include <Accelerate/Accelerate.h>

using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////
// Matrix Utilities

void mat_multiply_naive(MatrixF &A, const MatrixF &B, const MatrixF &C)
{
    const int rowsOut = B.extent(0);
    const int innerDim = B.extent(1);
    const int colsOut = C.extent(1);
    if (B.extent(1) != C.extent(0))
    {
        error_exit("Error: Inconsistent matrix dimensions! Exiting.");
    }
    if ((A.extent(0) != rowsOut) || (A.extent(1) != C.extent(1)))
    {
        A.resize(rowsOut, colsOut);
    }
    //#pragma omp parallel for
    for (int i = 0; i < rowsOut; i++)
    {
        for (int j = 0; j < colsOut; j++)
        {
            float sum = 0;
            for (int k = 0; k < innerDim; k++)
            {
                sum += B(i, k) * C(k, j);
            }
            A(i, j) = sum;
        }
    }
}

void mat_multiply(MatrixF &A, const MatrixF &B, const MatrixF &C)
{
    mat_multiply_blas(A, B, C); // Optimized BLAS version.
    // mat_multiply_naive(A, B, C); // Super slow naive version.
}

void mat_multiply(MatrixF &A, const MatrixF &B, const MatrixF &C, float alpha, float beta)
{
    mat_multiply_blas(A, B, C, alpha, beta);
}

// Use this if you have an optimized BLAS implementation (requires you include cblas.h)
void mat_multiply_blas(MatrixF &A, const MatrixF &B, const MatrixF &C)
{
    mat_multiply_blas(A, B, C, 1.0f, 0.0f);
}

void mat_multiply_blas(MatrixF &A, const MatrixF &B, const MatrixF &C, float alpha, float beta)
{
    if (B.extent(1) != C.extent(0))
    {
        error_exit("Error: Inconsistent matrix dimensions! Exiting.");
    }

    const int rows_A = B.extent(0);
    const int cols_A = C.extent(1);
    if ((A.order() != 2) || (A.size() != rows_A * cols_A))
    {
            A.resize(rows_A, cols_A);
    }

    float *backingArrayA = A.get_backing_data();
    const float *backingArrayB = B.get_backing_data();
    const float *backingArrayC = C.get_backing_data();

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.extent(0), A.extent(1), B.extent(1), alpha,
                backingArrayB, B.extent(1), backingArrayC, C.extent(1), beta, backingArrayA, A.extent(1));
}

void randomize_uniform(MatrixF &A, float min, float max)
{
    static std::random_device rand_dev;
    static std::mt19937 mersenne_twister_engine(rand_dev());
    // mersenne_twister_engine.seed(static_cast<unsigned long>(time(NULL)));
    std::uniform_real_distribution<float> uni(min, max);
    for (int i = 0; i < A.size(); i++)
    {
        A[i] = uni(mersenne_twister_engine);
    }
}

void assert_almost_equal_relative_error(const MatrixF &A, const MatrixF &B, float tolerance)
{
    check_dimensions(A, B);
    float score = relative_error(A, B);
    if (score > tolerance)
    {
        cerr << "Tolerance of " << tolerance << " was exceeded! error:" << score << endl;
        error_exit("");
    }
}

float assert_almost_equal_max_error(const MatrixF &A, const MatrixF &B, float tolerance) {
    float max_error = 0;
    float max_result_val = 0;
    for (int i = 0; i < A.size(); ++i)
    {
        const float error = std::abs(A[i] - B[i]);
        if (error > max_error) {
            max_error = error;
        }
        if (std::abs(A[i]) > max_result_val) {
            max_result_val = A[i];
        }
        if (error > tolerance) {
            cout << "Wrong result" << endl;
            cout << "Error amount: " << error << endl;
            cout << "A value: " << A[i] << endl;
            cout << "B value: " << B[i] << endl;
            error_exit("exiting");
        }
    }
    return max_error;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Tensor operations and math functions

float relative_error(const MatrixF &A, const MatrixF &B)
{
    if (A.size() != A.size())
    {
        error_exit("relative_error(): Must be same sizes.");
    }
    // Use method from: http://cs231n.github.io/neural-networks-3/
    //
    // |A - B|
    // ---------
    // max(|A|, |B|)
    //
    float numer = 0.0f; // L2_norm(a - b)
    float a_norm = 0.0f;
    float b_norm = 0.0f;
    for (int i = 0; i != A.size(); ++i)
    {
        numer += (A[i] - B[i]) * (A[i] - B[i]);
        a_norm += A[i] * A[i];
        b_norm += B[i] * B[i];
    }
    numer = sqrt(numer);
    a_norm = sqrt(a_norm);
    b_norm = sqrt(b_norm);
    if ((a_norm == 0) && (b_norm == 0))
    {
        return 0.0f;
    }
    else
    {
        float res = numer / std::max(a_norm, b_norm);
        return res;
    }
}

void compute_forward_relu_in_place(MatrixF &x)
    {
        //#pragma omp parallel for
        for (int n = 0; n < x.size(); ++n)
        {
            if (x[n] <= 0.0f) {
                x[n] = 1e-9f; // Don't use exactly 0 for performance reasons
            }
        }
    }
