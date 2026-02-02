import numpy as np
import matplotlib.pyplot as plt

def load_sets(filename):
    sets = []
    current = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                if current:
                    sets.append(np.array(current))
                    current = []
            else:
                vals = list(map(float, line.split()))

                # file format: t x y
                if len(vals) >= 3:
                    _, x, y = vals[:3]
                    current.append([x, y])

    if current:
        sets.append(np.array(current))

    return sets


# ===== main =====
sets = load_sets("ex6_2_2.dat")

plt.figure(figsize=(7, 7))

for k, S in enumerate(sets):
    plt.plot(S[:, 0], S[:, 1], ".", markersize=4, label=f"set {k}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Exercise 6.2.2 – Artsy Set Propagation")
plt.axis("equal")
plt.grid(True)
plt.legend()

plt.show()

