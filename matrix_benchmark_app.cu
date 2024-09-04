// matrix_benchmark_app.cu
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iomanip>

__global__ void matrixMultiply(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void matrixAdd(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        C[row * n + col] = A[row * n + col] + B[row * n + col];
    }
}

// CUDA kernel for inverting a matrix using Gaussian elimination
__global__ void matrixInvert(float *A, float *I, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = 0; i < n; i++) {
        if (tid == i) {
            // Divide the current row by the diagonal element
            float diag = A[i * n + i];
            for (int j = 0; j < n; j++) {
                A[i * n + j] /= diag;
                I[i * n + j] /= diag;
            }
        }
        __syncthreads();

        // Eliminate other rows
        for (int j = 0; j < n; j++) {
            if (tid != i && tid == j) {
                float factor = A[j * n + i];
                for (int k = 0; k < n; k++) {
                    A[j * n + k] -= factor * A[i * n + k];
                    I[j * n + k] -= factor * I[i * n + k];
                }
            }
            __syncthreads();
        }
    }
}

void matrixMultiplyCPU(float *A, float *B, float *C, int n) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            float sum = 0.0f;
            for (int i = 0; i < n; i++) {
                sum += A[row * n + i] * B[i * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}

void matrixAddCPU(float *A, float *B, float *C, int n) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            C[row * n + col] = A[row * n + col] + B[row * n + col];
        }
    }
}

// CPU implementation of matrix inversion using Gaussian elimination
void matrixInvertCPU(float *A, float *I, int n) {
    for (int i = 0; i < n; i++) {
        // Divide the current row by the diagonal element
        float diag = A[i * n + i];
        for (int j = 0; j < n; j++) {
            A[i * n + j] /= diag;
            I[i * n + j] /= diag;
        }

        // Eliminate other rows
        for (int j = 0; j < n; j++) {
            if (i != j) {
                float factor = A[j * n + i];
                for (int k = 0; k < n; k++) {
                    A[j * n + k] -= factor * A[i * n + k];
                    I[j * n + k] -= factor * I[i * n + k];
                }
            }
        }
    }
}

int main() {
    int N;
    int operation_choice;

    // Prompt user for matrix size
    std::cout << "Enter matrix size (N for NxN matrix): ";
    std::cin >> N;

    if (N <= 0) {
        std::cerr << "Matrix size must be a positive integer." << std::endl;
        return 1;
    }

    // Prompt user for operation choice
    std::cout << "Choose the operation:\n";
    std::cout << "1. Matrix Multiplication\n";
    std::cout << "2. Matrix Addition\n";
    std::cout << "3. Matrix Inversion\n";
    std::cout << "Enter your choice (1, 2, or 3): ";
    std::cin >> operation_choice;

    if (operation_choice < 1 || operation_choice > 3) {
        std::cerr << "Invalid choice. Please select 1, 2, or 3." << std::endl;
        return 1;
    }

    int size = N * N * sizeof(float);

    // Allocate memory on host
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_gpu = (float *)malloc(size);
    float *h_C_cpu = (float *)malloc(size);
    float *h_I = (float *)malloc(size); // Identity matrix for inversion

    // Initialize matrices with random values
    for (int i = 0; i < N * N; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        h_I[i] = (i / N == i % N) ? 1.0f : 0.0f; // Set identity matrix
    }

    // Allocate memory on device
    float *d_A, *d_B, *d_C, *d_I;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMalloc(&d_I, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_I, h_I, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    std::chrono::duration<double> cpu_time, gpu_time;

    // Perform the selected operation and measure time
    if (operation_choice == 1) {
        // Matrix Multiplication
        auto start_gpu = std::chrono::high_resolution_clock::now();
        matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
        auto end_gpu = std::chrono::high_resolution_clock::now();
        gpu_time = end_gpu - start_gpu;

        auto start_cpu = std::chrono::high_resolution_clock::now();
        matrixMultiplyCPU(h_A, h_B, h_C_cpu, N);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        cpu_time = end_cpu - start_cpu;

    } else if (operation_choice == 2) {
        // Matrix Addition
        auto start_gpu = std::chrono::high_resolution_clock::now();
        matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
        auto end_gpu = std::chrono::high_resolution_clock::now();
        gpu_time = end_gpu - start_gpu;

        auto start_cpu = std::chrono::high_resolution_clock::now();
        matrixAddCPU(h_A, h_B, h_C_cpu, N);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        cpu_time = end_cpu - start_cpu;

    } else if (operation_choice == 3) {
        // Matrix Inversion
        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        auto start_gpu = std::chrono::high_resolution_clock::now();
        matrixInvert<<<blocks, threads>>>(d_A, d_I, N);
        cudaDeviceSynchronize();
        auto end_gpu = std::chrono::high_resolution_clock::now();
        gpu_time = end_gpu - start_gpu;

        auto start_cpu = std::chrono::high_resolution_clock::now();
        matrixInvertCPU(h_A, h_I, N);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        cpu_time = end_cpu - start_cpu;
    }

    // Copy result back to host
    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    // Save results to file
    std::ofstream results_file("timing_results.txt", std::ios::app);  // Append mode
    results_file << "Matrix Size: " << N << " x " << N << std::endl;
    if (operation_choice == 1) {
        results_file << "Operation: Matrix Multiplication" << std::endl;
    } else if (operation_choice == 2) {
        results_file << "Operation: Matrix Addition" << std::endl;
    } else {
        results_file << "Operation: Matrix Inversion" << std::endl;
    }
    results_file << "CPU Time (seconds): " << cpu_time.count() << std::endl;
    results_file << "GPU Time (seconds): " << gpu_time.count() << std::endl;
    results_file << "Speedup (CPU/GPU): " << cpu_time.count() / gpu_time.count() << std::endl;
    results_file << "------------------------" << std::endl;
    results_file.close();

    // Output results to console
    std::cout << "CPU Time: " << cpu_time.count() << " seconds\n";
    std::cout << "GPU Time: " << gpu_time.count() << " seconds\n";
    std::cout << "Speedup (CPU/GPU): " << cpu_time.count() / gpu_time.count() << "x\n";

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_I);
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);
    free(h_I);

    return 0;
}