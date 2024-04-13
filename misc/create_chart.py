import matplotlib.pyplot as plt

# Data from the LaTeX table
methods = ["Gaussian", "Sinusoid", "Wavelet", "Sinc"]
n_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
data = {
    "Gaussian": [31.13, 0.947, 31.13, 0.947, 31.13, 0.947, 31.13, 0.947, 31.13, 0.947],
    "Sinusoid": [28.96, 0.933, 31.13, 0.947, 31.13, 0.947, 31.13, 0.947, 31.13, 0.947],
    "Wavelet": [30.33, 0.941, 31.13, 0.947, 31.13, 0.947, 31.13, 0.947, 31.13, 0.947],
    "Sinc": [31.37, 0.947, 31.13, 0.947, 31.13, 0.947, 31.13, 0.947, 31.13, 0.947],
}

# Create a line chart for each method
for method in methods:
    plt.plot(n_values, data[method], label=method)

# Set labels and title
plt.xlabel('n')
plt.ylabel('Values')
plt.title('Line Chart of Values for Different Methods')

# Add a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.savefig(f"eye_catcherso/energy_mat.png")
plt.close()
