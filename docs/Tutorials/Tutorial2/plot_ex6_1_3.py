import numpy as np
import matplotlib.pyplot as plt

filename = "ex6_1_3.dat"

# --- read data and split by time blocks ---
blocks = []
current_block = []

with open(filename, "r") as f:
    for line in f:
        if line.strip() == "":
            if current_block:
                blocks.append(np.array(current_block))
                current_block = []
        else:
            t, x, y = map(float, line.split())
            current_block.append([t, x, y])

if current_block:
    blocks.append(np.array(current_block))

# --- plot ---
plt.figure(figsize=(7, 7))

for k, block in enumerate(blocks):
    x = block[:, 1]
    y = block[:, 2]
    plt.plot(x, y, marker="o", markersize=3, linestyle="-",
             label=f"t = {block[0,0]:.2f}")

plt.xlabel("x₁")
plt.ylabel("x₂")
plt.title("Exercise 6.1.3 – Set propagation")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
