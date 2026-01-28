import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Load data
# Columns:
# 0 -> M (rad)
# 1 -> E from DA
# 2 -> E from pointwise Newton
data = np.loadtxt("ex4_2_2.dat")

M = data[:, 0]
E_DA = data[:, 1]
E_pw = data[:, 2]

# Convert radians to degrees
M_deg = M * 180.0 / pi
E_DA_deg = E_DA * 180.0 / pi
E_pw_deg = E_pw * 180.0 / pi

# Plot
plt.figure()
plt.plot(M_deg, E_DA_deg, label="DA", linewidth=2)
plt.plot(M_deg, E_pw_deg, "--", label="Pointwise Newton", linewidth=2)
plt.xlabel("Mean anomaly M [deg]")
plt.ylabel("Eccentric anomaly E [deg]")
plt.title("Kepler equation solution: DA vs pointwise")
plt.legend()
plt.grid()
plt.show()
