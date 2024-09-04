#!/bin/bash

# Define matrix sizes to test
MATRIX_SIZES=(128 256 512)

# Define operations (1 = Multiplication, 2 = Addition, 3 = Inversion)
OPERATIONS=(1 2 3)

# Compile the CUDA application
echo "Compiling matrix_benchmark_app.cu..."
nvcc matrix_benchmark_app.cu -o matrix_benchmark_app
if [ $? -ne 0 ]; then
    echo "Compilation failed. Exiting."
    exit 1
fi
echo "Compilation successful."

# Function to run the benchmark for a given size and operation
run_benchmark() {
    size=$1
    operation=$2
    echo "Running benchmark for Matrix Size: $size, Operation: $operation"
    
    # Pass the size and operation to the executable via input redirection
    echo -e "$size\n$operation" | ./matrix_benchmark_app
}

# Iterate over all matrix sizes, operations, and run benchmarks
for size in "${MATRIX_SIZES[@]}"; do
    for operation in "${OPERATIONS[@]}"; do
        # Run benchmark for the GPU
        run_benchmark $size $operation

        # To simulate CPU-only runs, you can set a flag within your application
        # or handle the distinction via the program inputs
        # Here, it assumes the CPU/GPU distinction is handled by the matrix_benchmark_app
    done
done

echo "All benchmarks completed."