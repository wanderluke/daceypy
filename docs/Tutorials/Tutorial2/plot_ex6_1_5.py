import numpy as np
import matplotlib.pyplot as plt

# load data (skip the first dummy line and blank lines)
data = np.loadtxt("ex6_1_5.dat")

x = data[:, 0]
y = data[:, 1]

plt.figure(figsize=(6, 6))
plt.plot(x, y, "-o", markersize=4)
plt.scatter([1.0], [1.0], color="red", label="Initial point (1,1)")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Final state after 2π as function of parameter α")
plt.axis("equal")
plt.grid(True)
plt.legend()

plt.show()
