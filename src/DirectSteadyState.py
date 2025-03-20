import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from matplotlib.colors import LogNorm
from scipy.sparse.linalg import spsolve
from EigenmodesTimeIndependent import laplacian_circular

N = 8
radius = 2

# Construct the Laplacian matrix
M, _, _, mask = laplacian_circular(N, radius, 2, 2)

# Set a source on point (0.6, 1.2)
source = (0.6, 1.2)
i, j = int(-(source[1] + radius) / (2 * radius) * N), int((source[0] + radius) / (2 * radius) * N)
source_index = i * N + j

# Set the source to 1
M[source_index, :] = 0
M[source_index, source_index] = 1
b = np.zeros(N**2)
b[source_index] = 1

# Plot the Laplacian matrix and the mask next to each other
# plt.figure(figsize=(12,6))
plt.subplots(1, 2)
plt.subplot(1, 2, 1)
plt.imshow(M.toarray(), cmap='viridis')
plt.colorbar(label="Value")
plt.xlabel("Column")
plt.ylabel("Row")
plt.title("Laplacian Matrix M")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='viridis')
plt.colorbar(label="Mask Value")
plt.xlabel("Column")
plt.ylabel("Row")
plt.title("Mask")
plt.show()

M = M.tocsr()
# Solve the system
x = spsolve(M, b)

# Plot the solution with log scale
plt.figure(figsize=(6, 5))
plt.imshow(x.reshape(N, N), cmap='viridis', extent=(-radius, radius, -radius, radius),
           norm=LogNorm(vmin=np.max(x) * 1e-3, vmax=np.max(x)))
plt.colorbar(label="Concentration (log scale)")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Solution (Log Scale)")
plt.show()
