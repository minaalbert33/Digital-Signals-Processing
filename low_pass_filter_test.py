import numpy as np
import matplotlib.pyplot as plt




def manual_convolve(x, h): # Question 2
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


def generate_complex_test_signal(samples_number=500, fs=500, f1_cycles=6, A1=1, f2_cycles=44, A2=0.5):
    """Generate a complex test signal consisting of two sinusoids."""
    n = np.arange(samples_number)
    
    # first sinusoid
    f1 = f1_cycles / samples_number * fs
    sinusoid1 = A1 * np.sin(2 * np.pi * f1 * n)

    # second sinusoid
    f2 = f2_cycles / samples_number * fs
    sinusoid2 = A2 * np.sin(2 * np.pi * f2 * n)

    plt.figure(figsize=(10, 6))
    plt.plot(n, sinusoid2, label='Test Signal $x[n]$')
    plt.title('Test Signal with Two Sinusoids')
    plt.xlabel('n')
    plt.ylabel('$x[n]$')
    plt.grid(True)
    plt.legend()
    plt.show()
    x = sinusoid1 + sinusoid2
    
    return x, n


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

    def plot_impulse_response(self): # Question 1
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
    """Question 1"""
    # filter.plot_impulse_response()

    """Question 2"""
    y = filter.apply_filter(x)
    print(y)
    # plt.stem(y)
    # plt.title('Filtered Signal y[n]')
    # plt.xlabel('n')
    # plt.ylabel('y[n]')
    # plt.grid(True)
    # plt.show()

    """Question 3"""
    complicated_signal, n = generate_complex_test_signal()
    print(complicated_signal)

    # plt.stem(complicated_signal)
    # plt.title('Combined Signal x[n]')
    # plt.xlabel('n')
    # plt.ylabel('x[n]')
    # plt.grid(True)
    # plt.show()

    """Question 4"""
    complicated_signal_filtered = filter.apply_filter(complicated_signal)

    plt.stem(complicated_signal_filtered)
    plt.title('Combined Signal filtered x[n]')
    plt.xlabel('n')
    plt.ylabel('x[n]')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

