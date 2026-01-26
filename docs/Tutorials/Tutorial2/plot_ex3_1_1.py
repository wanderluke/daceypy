import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("ex3_1_1.dat")

x = data[:, 0]
da = data[:, 1]
true = data[:, 2]

# Plot DA vs exact
plt.figure()
plt.plot(x, da, label="DA approximation")
plt.plot(x, true, label="Exact function")
plt.xlabel("x")
plt.ylabel("value")
plt.legend()
plt.grid()
plt.title("ex3_1_1: DA vs Exact")
plt.show()

# Plot error
plt.figure()
plt.plot(x, da - true)
plt.xlabel("x")
plt.ylabel("DA - Exact")
plt.title("Error")
plt.grid()
plt.show()
