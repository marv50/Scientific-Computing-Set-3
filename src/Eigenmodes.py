import numpy as np
import scipy.sparse as sp
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
    side_diag[np.arange(1, size) % N == 0] = 0  # Remove wrap-around connections
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

# Example for N = 4
N = 4
M = laplacian_2d_diagonal(N)

# Convert to dense for small N (for visualization)
print(M.toarray())

# Show as image
plt.imshow(M.toarray())
plt.show()