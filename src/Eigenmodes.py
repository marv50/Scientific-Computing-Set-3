import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib as mpl
import matplotlib.pyplot as plt


def laplacian_2d_diagonal(N):
    """Constructs the 2D Laplacian matrix using diagonal representation."""
    size = N * N  # Total number of unknowns

    # Main diagonal (-4)
    main_diag = -4 * np.ones(size)

    # Off-diagonals (+1 for nearest neighbors)
    side_diag = np.ones(size - 1)
    print(side_diag)
    # Remove wrap-around connections
    side_diag[np.arange(1, size) % N == 0] = 0
    print(side_diag)

    up_down_diag = np.ones(size - N)  # For vertical connections
    print(up_down_diag)

    # Construct sparse matrix using diagonals
    M = sp.diags(
        [main_diag, side_diag, side_diag, up_down_diag, up_down_diag],
        [0, -1, 1, -N, N],  # Diagonal offsets
        shape=(size, size),
        format="csr"
    )

    return M


"""
TO DO: Implement a way to simulate on circular or rectangular domains. 
Possibly by adjusting our matrix M.
"""

# Example for N = 4
N = 4
M = laplacian_2d_diagonal(N)

eigenvalues, eigenvectors = spla.eigs(
    M, k=5, which="SM")  # Smallest 5 eigenvalues

v1 = eigenvectors[:, 0].real.reshape((N, N))  # Reshape back to 2D grid

print(len(eigenvectors))

# Show matrix as image
plt.title("Laplacian Matrix Vizualization")
plt.imshow(M.toarray())
plt.colorbar(label="Matrix Value")
plt.show()

# Plot first eigenvector
plt.imshow(v1, cmap="copper", extent=[0, N, 0, N])
plt.colorbar(label="Amplitude")
plt.title("First Eigenmode")
plt.show()
