import matplotlib.pyplot as plt
import numpy as np

# Read timing results from file
file_path = 'timing_results.txt'
matrix_sizes = []
cpu_times = []
gpu_times = []
operations = []

# Parse the file to extract data
with open(file_path, 'r') as file:
    lines = file.readlines()
    for i in range(0, len(lines), 6):  # Each result block is 6 lines long
        matrix_size = lines[i].strip().split(": ")[1]
        operation = lines[i + 1].strip().split(": ")[1]
        cpu_time = float(lines[i + 2].strip().split(": ")[1])
        gpu_time = float(lines[i + 3].strip().split(": ")[1])

        matrix_sizes.append(matrix_size)
        cpu_times.append(cpu_time)
        gpu_times.append(gpu_time)
        operations.append(operation)

# Unique operations and matrix sizes
unique_operations = list(set(operations))
unique_sizes = list(set(matrix_sizes))

# Sort operations and matrix sizes for consistent plotting
unique_operations.sort()
unique_sizes.sort()

# Prepare data for plotting
plot_data = {op: {'cpu': [], 'gpu': [], 'sizes': []} for op in unique_operations}

for i, op in enumerate(operations):
    plot_data[op]['cpu'].append(cpu_times[i])
    plot_data[op]['gpu'].append(gpu_times[i])
    plot_data[op]['sizes'].append(matrix_sizes[i])

cpu_color = '#76B900'
gpu_color = '#4D4D4D'

# Plotting Execution Times for each operation
for operation in unique_operations:
    sizes = plot_data[operation]['sizes']
    cpu = plot_data[operation]['cpu']
    gpu = plot_data[operation]['gpu']

    x = np.arange(len(sizes))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, cpu, width, label='CPU Time', color=cpu_color)
    rects2 = ax.bar(x + width/2, gpu, width, label='GPU Time', color=gpu_color)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Matrix Sizes')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title(f'Execution Time Comparison - {operation}')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend(loc='upper left')

    # Attach a text label above each bar in *rects*, displaying its height.
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    # Save the figure with a descriptive filename
    filename = f'execution_time_comparison_{operation.replace(" ", "_")}.png'
    fig.savefig(filename)
    print(f'Saved plot as {filename}')
