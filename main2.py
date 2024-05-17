import numpy as np
import matplotlib.pyplot as plt




def manual_convolve(x, h):
    """Convolve two signals x and h"""
    N = len(x)
    M = len(h)
    # full convolution (N + M - 1)
    y = np.zeros(N + M - 1)
    
    # Perform the convolution
    for n in range(N + M - 1):
        for k in range(N):
            if 0 <= n - k < M:
                y[n] += x[k] * h[n - k]
    
    return y

class LowPassWindowedSincFilter:
    def __init__(self, N=99, cutoff_cycles=25, total_samples=500):
        self.N = N  # Length of the filter
        self.cutoff_cycles = cutoff_cycles  
        self.total_samples = total_samples  
        self.fc = 2 * np.pi * (cutoff_cycles / total_samples)  
        self.alpha = (N - 1) / 2  # Midpoint of the filter
        self.h = np.zeros(N)  
        self._generate_impulse_response()

    def _generate_impulse_response(self): # private func
        for i in range(self.N):
            if i == self.alpha:
                self.h[i] = 0.31752 * self.fc  
            else:
                self.h[i] = 0.31752 * np.sin(self.fc * (i - self.alpha)) / (i - self.alpha)
            self.h[i] = self.h[i] * (0.54 - 0.46 * np.cos(2 * np.pi * i / (self.N - 1)))

    def apply_filter(self, x):
        return manual_convolve(x, self.h)

    def get_impulse_response(self):
        return self.h

    def plot_impulse_response(self):
        plt.stem(self.h)
        plt.title('Impulse Response h[n]')
        plt.xlabel('n')
        plt.ylabel('h[n]')
        plt.grid(True)
        plt.show()

def main():

    filter = LowPassWindowedSincFilter()

    #x[n] = 1 for n = 0 , x[n] = 0 otherwise
    x = np.zeros(500)
    x[0] = 1

    # y[n] = x[n] * h[n]
    y = filter.apply_filter(x)

    print(y)
    # filter.plot_impulse_response()
    plt.stem(x)
    plt.title('Filtered Signal y[n]')
    plt.xlabel('n')
    plt.ylabel('y[n]')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

