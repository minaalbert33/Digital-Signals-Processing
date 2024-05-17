import numpy as np
import matplotlib.pyplot as plt
# # for generating the same random numbers
# np.random.seed(42)
Nx = 500
# x = np.random.randn(500)  # Example x[n] with 500 samples
# h = np.random.randn(99)   # Example h[n] with 99 samples
#
# # Calculate the convolution y[n]
# y = np.convolve(x, h)
# print(y[1])
#

N = 99  # Length of the filter
cycles_per_sample = 25 / 500        
cutoff_frequency = 2 * np.pi * cycles_per_sample 

# Midpoint of the filter
alpha = (N - 1) / 2                         

# Initialize the h[n] array
h = np.zeros(N) 

# get hte Impulse Response h[n]
for i in range(N):
    if i == alpha:
        h[i] = 0.31752 * cutoff_frequency  # Handle the division by zero case
    else:
        h[i] = 0.31752 * np.sin(cutoff_frequency * (i - alpha)) / (i - alpha)
    h[i] = h[i] * (0.54 - 0.46 * np.cos(0.0641114 * i))


print(h)
# Plot the impulse response
# plt.stem(h)
# plt.title('Impulse Response h[n]')
# plt.xlabel('n')
# plt.ylabel('h[n]')
# plt.grid(True)
# plt.show()
#
