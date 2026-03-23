# 1) Take a random number
# 2) Multiply it by 2 or divide it by 2 with the same probability
# 3) Iterate the process many times (100)
# 4) Run with many different seeds
# 5) Compute the histograms with the output

import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 1000  # Number of iterations for each seed
seeds = np.random.randint(0, 200, size=1000)
results = []

for seed in seeds:
    x = seed
    for i in range(N):
        if np.random.rand() < 0.5:
            x *= 2
        else:
            x /= 2
    results.append(x)

# The histogram will be the frequency of the first digit of the results
first_digits = [str(int(abs(result)))[0] for result in results if result != 0]
digit_counts = {str(i): first_digits.count(str(i)) for i in range(1, 10)}

# Plotting the histogram
plt.bar(digit_counts.keys(), digit_counts.values())
plt.xlabel('First Digit')
plt.ylabel('Frequency')
plt.title('Histogram of First Digits')
plt.show()