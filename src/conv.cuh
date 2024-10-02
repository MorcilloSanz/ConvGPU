/**
 * @file conv.cuh
 * @author Alberto Morcillo Sanz (amorcillosanz@gmail.com)
 * @brief Provides implementations for 2D convolution operations using CUDA.
 * 
 * This header file defines the necessary structures and functions to perform 
 * 2D discrete convolution on matrices utilizing CUDA for parallel processing.
 * It includes a matrix class template for managing 2D data and functions for 
 * performing convolution, error checking, and other related operations.
 * 
 * @version 0.1
 * @date 2024-10-02
 * 
 * @copyright Copyright (c) 2024 Alberto Morcillo Sanz. All rights reserved.
 */

#pragma once

#include <iostream>
#include <initializer_list>
#include <algorithm> 

#include <cuda_runtime.h>

namespace cnv
{

/**
 * @brief Basic Matrix struct for computing 2D convolutions.
 * 
 * This class template provides a basic implementation of a 2D matrix
 * with functionalities for element access, assignment, and copying.
 * It supports construction from dimensions, an initializer list, 
 * copy/move operations, and printing.
 * 
 * @tparam T Type of elements stored in the matrix.
 */
template <typename T>
struct Matrix 
{
    /**
     * @brief Pointer to the data array storing matrix elements.
     */
    T* data;

    /**
     * @brief Number of rows in the matrix.
     */
    unsigned int rows;

    /**
     * @brief Number of columns in the matrix.
     */
    unsigned int cols;

    /**
     * @brief Constructs a matrix with specified rows and columns.
     * 
     * @param _rows Number of rows.
     * @param _cols Number of columns.
     */
    Matrix(unsigned int _rows, unsigned int _cols)
        : rows(_rows), cols(_cols) {
        data = new T[rows * cols];
    }

    /**
     * @brief Copy constructor.
     * 
     * Constructs a new matrix by copying data from an existing matrix.
     * 
     * @param matrix The matrix to copy from.
     */
    Matrix(const Matrix& matrix)
        : rows(matrix.rows), cols(matrix.cols) {
        data = new T[rows * cols];
        std::copy(matrix.data, matrix.data + rows * cols, data);
    }

    /**
     * @brief Move constructor.
     * 
     * Constructs a new matrix by transferring ownership of the data 
     * from an existing matrix.
     * 
     * @param matrix The matrix to move from.
     */
    Matrix(Matrix&& matrix) noexcept
        : data(matrix.data), rows(matrix.rows), cols(matrix.cols) {
        matrix.data = nullptr;
    }

    /**
     * @brief Default constructor.
     * 
     * Constructs an empty matrix.
     */
    Matrix() = default;

    /**
     * @brief Destructor.
     * 
     * Frees the memory allocated for the matrix data.
     */
    ~Matrix() {
        delete[] data;
    }

    /**
     * @brief Copy assignment operator.
     * 
     * Assigns the data of one matrix to another by copying.
     * 
     * @param matrix The matrix to copy from.
     * @return Reference to the assigned matrix.
     */
    Matrix& operator=(const Matrix& matrix) {

        if (this != &matrix) {
            delete[] data;
            rows = matrix.rows;
            cols = matrix.cols;
            data = new T[rows * cols];
            std::copy(matrix.data, matrix.data + rows * cols, data);
        }

        return *this;
    }

    /**
     * @brief Move assignment operator.
     * 
     * Transfers ownership of the data from one matrix to another.
     * 
     * @param matrix The matrix to move from.
     * @return Reference to the assigned matrix.
     */
    Matrix& operator=(Matrix&& matrix) noexcept {

        if (this != &matrix) {
            delete[] data;
            data = matrix.data;
            rows = matrix.rows;
            cols = matrix.cols;
            matrix.data = nullptr;
        }

        return *this;
    }

    /**
     * @brief Constructs a matrix from an initializer list.
     * 
     * @param list An initializer list of initializer lists representing
     * the rows and elements of the matrix.
     */
    Matrix(const std::initializer_list<std::initializer_list<T>>& list) {

        rows = list.size();
        cols = list.begin()->size();
        data = new T[rows * cols];

        unsigned int i = 0;
        for(const auto& row : list) {
            
            unsigned int j = 0;
            for(const auto& elem : row) {
                set(elem, i, j);
                j ++;
            }

            i ++;
        }
    }

    /**
     * @brief Sets the value of an element in the matrix.
     * 
     * @param value The value to set.
     * @param i Row index.
     * @param j Column index.
     */
    inline void set(const T& value, unsigned int i, unsigned int j) {
        data[i + j * cols] = value;
    }

    /**
     * @brief Gets the value of an element in the matrix.
     * 
     * @param i Row index.
     * @param j Column index.
     * @return The value at the specified row and column.
     */
    inline T get(unsigned int i, unsigned int j) const {
        return data[i + j * cols];
    }

    /**
     * @brief Outputs the matrix to an output stream.
     * 
     * @param os The output stream.
     * @param matrix The matrix to output.
     * @return The output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {

        for(int i = 0; i < matrix.cols; i ++) {
            for(int j = 0; j < matrix.rows; j ++)
                os << matrix.get(i, j) << " ";
            os << "\n";
        }

        return os;
    }
};

using Kernel = Matrix<float>;

/**
 * @brief Checks for errors from previously executed CUDA functions and reports them.
 * 
 * This function verifies whether any CUDA function called prior to this check has 
 * returned an error. If an error is detected, it prints the provided user message 
 * along with the error description, aiding in debugging.
 * 
 * @param message A user-defined message to provide context about where the error check is being performed.
 */
void check_cuda_error(const char* message) {

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        std::cerr << "CUDA error after " << message << ": " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

/**
 * @brief Performs a 2D discrete convolution on an input matrix.
 * 
 * This function computes the 2D discrete convolution of an input matrix
 * with a given kernel, storing the result in the output matrix. The 
 * function is designed to run on a GPU using CUDA.
 * 
 * @tparam T Type of the elements in the input, output, and kernel matrices.
 * @param input Pointer to the input matrix (flattened as a 1D array).
 * @param output Pointer to the output matrix (flattened as a 1D array) where the convolution result will be stored.
 * @param kernel Pointer to the kernel matrix (flattened as a 1D array).
 * @param width Width of the input matrix (number of columns).
 * @param height Height of the input matrix (number of rows).
 * @param kernelWidth Width of the kernel matrix (number of columns).
 * @param kernelHeight Height of the kernel matrix (number of rows).
 */
template <typename T>
__global__ void kernel_conv2D(T* input, T* output, T* kernel, unsigned int width, 
    unsigned int height, unsigned int kernelWidth, unsigned int kernelHeight) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) 
        return;

    int kx = kernelWidth / 2;
    int ky = kernelHeight / 2;

    T sum = 0;
    for(int i = -kx; i <= kx; i ++) {
        for(int j = -ky; j <= ky; j ++) {

            int xi = x + i;
            int yj = y + j;

            T value = 0;
            if(xi < width && xi >= 0 && yj < height && yj >= 0)
                value = input[xi + yj * width];

            sum += value * kernel[(i + kx) + (j + ky) * kernelHeight];
        }
    }

    output[y * width + x] = sum;
}

/**
 * @brief Performs a 2D discrete convolution on an input matrix.
 * 
 * This function computes the 2D discrete convolution of an input matrix
 * with a given kernel, storing the result in the output matrix. The 
 * function is designed to run on a GPU using CUDA.
 * 
 * @tparam T 
 * @param input input matrix
 * @param output output matrix
 * @param kernel kernel matrix
 */
template <typename T>
void conv2D(const Matrix<T>& input, Matrix<T>& output, const Kernel& kernel) {

    const unsigned int width = input.cols;
    const unsigned int height = input.rows;
    const unsigned int kernelWidth = kernel.cols;
    const unsigned int kernelHeight = kernel.rows;

    T *d_input, *d_output, *d_kernel;
    T size = width * height * sizeof(T);
    T sizeKernel = kernelWidth * kernelHeight * sizeof(int);

    // Reserve memory
    cudaMalloc((void **)&d_input, size);
    check_cuda_error("cudaMalloc d_input");

    cudaMalloc((void **)&d_output, size);
    check_cuda_error("cudaMalloc d_input");

    cudaMalloc((void **)&d_kernel, sizeKernel);
    check_cuda_error("cudaMalloc d_kernel");

    // Copy input to GPU
    cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice);
    check_cuda_error("cudaMemcpy d_input");

    cudaMemcpy(d_kernel, kernel.data, sizeKernel, cudaMemcpyHostToDevice);
    check_cuda_error("cudaMemcpy d_kernel");

    // Call kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    kernel_conv2D<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_kernel, width, height, kernelWidth, kernelHeight);
    cudaDeviceSynchronize();

    // Copy output to CPU
    cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

}