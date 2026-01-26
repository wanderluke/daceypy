import numpy as np
import matplotlib.pyplot as plt

# Load data
# Columns:
# 0 -> computation order
# 1 -> integral value
# 2 -> log10(error)
data = np.loadtxt("ex3_1_4.dat")

order = data[:, 0]
value = data[:, 1]
logerr = data[:, 2]

# --- Plot integral value vs order ---
plt.figure()
plt.plot(order, value, marker="o")
plt.xlabel("DA order")
plt.ylabel("Integral value")
plt.title("Gaussian integral on [-1,1]")
plt.grid()
plt.show()

# --- Plot log10(error) vs order ---
plt.figure()
plt.plot(order, logerr, marker="o")
plt.xlabel("DA order")
plt.ylabel("log10(error)")
plt.title("Convergence of DA approximation")
plt.grid()
plt.show()
