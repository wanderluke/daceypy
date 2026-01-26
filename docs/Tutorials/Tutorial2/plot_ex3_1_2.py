import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Load data
data = np.loadtxt("ex3_1_2.dat")

# Columns:
# 0 -> dx
# 1 -> dy
# 2 -> DA value
# 3 -> exact value
x = data[:, 0]
y = data[:, 1]
da = data[:, 2]
true = data[:, 3]
err = da - true

# --- DA surface ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot_trisurf(x, y, da, cmap="viridis")
ax.set_title("DA approximation")
ax.set_xlabel("dx")
ax.set_ylabel("dy")
ax.set_zlabel("value")
plt.tight_layout()
plt.show()

# --- Exact surface ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot_trisurf(x, y, true, cmap="viridis")
ax.set_title("Exact function")
ax.set_xlabel("dx")
ax.set_ylabel("dy")
ax.set_zlabel("value")
plt.tight_layout()
plt.show()

# --- Error surface ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot_trisurf(x, y, err, cmap="coolwarm")
ax.set_title("DA error (DA - exact)")
ax.set_xlabel("dx")
ax.set_ylabel("dy")
ax.set_zlabel("error")
plt.tight_layout()
plt.show()
