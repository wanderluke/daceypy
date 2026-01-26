import numpy as np
import matplotlib.pyplot as plt

# Load data
# Columns:
# 0 -> number of subintervals n
# 1 -> integral value
# 2 -> log10(error)
data = np.loadtxt("ex3_2_1.dat")

n = data[:, 0]
value = data[:, 1]
logerr = data[:, 2]

# --- Plot log10(error) vs number of subintervals ---
plt.figure()
plt.plot(n, logerr, marker="o", linestyle="-")
plt.xlabel("Number of subintervals (n)")
plt.ylabel("log10(error)")
plt.title("Gaussian integral on [-2,2]\nDA order = 9, domain splitting")
plt.grid()
plt.show()

# --- Optional: plot integral value vs n ---
plt.figure()
plt.plot(n, value, marker="o", linestyle="-")
plt.xlabel("Number of subintervals (n)")
plt.ylabel("Integral value")
plt.title("Integral convergence with domain splitting")
plt.grid()
plt.show()
