import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def create_laplacian_matrix(N, h, domain="square"):
    """Creates the discrete Laplacian matrix for a 2D grid with Dirichlet BCs."""
    
    size = N**2
    
    diag_main = -4 * np.ones(size) / h**2
    diag_x = np.ones(size) / h**2
    diag_y = np.ones(size) / h**2

    # Construct sparse Laplacian matrix using 5-point stencil
    A = sp.diags([diag_main, diag_x[:-1], diag_x[:-1], diag_y[:-N], diag_y[:-N]], 
                 [0, 1, -1, N, -N], shape=(size, size), format="csr")

    # Remove incorrect connections at right boundaries
    for i in range(1, N):
        A[i * N - 1, i * N] = 0  # Remove right boundary connections
        A[i * N, i * N - 1] = 0  # Remove left boundary connections

    return A

# Parameters
L = 1.0   # Domain size (1x1 square)
h = 0.1   # Step size
N = 300  # Number of interior grid points

# Create the Laplacian matrix with Dirichlet BCs
A = create_laplacian_matrix(N, h, "square")

# Eigenvalue problem: solve A v = K v
eigenvalues, eigenvectors = spla.eigs(A, k=5, which="SM")  # Smallest 5 eigenvalues

# Print Results
print("Laplacian Matrix (Sparse Format):")
print(A)
print("\nFirst 5 Eigenvalues:", eigenvalues.real)

# Plot first eigenvector
v1 = eigenvectors[:, 0].real.reshape((N, N))  # Reshape back to 2D grid
plt.imshow(v1, cmap="copper", extent=[0, L, 0, L])
plt.colorbar()
plt.show()
