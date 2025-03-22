import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.sparse.linalg import spsolve
if __name__ == '__main__':
    from EigenmodesTimeIndependent import laplacian_circular
else:
    from src.EigenmodesTimeIndependent import laplacian_circular

def solve_steady_state(N: int, radius: float, source:tuple = (0.6, 1.2), plot_M_mask:bool = False) -> np.ndarray:
    """
    Solves the steady state heat equation for a circular domain with a source at a given point.
    The source is a point source, i.e. the source is a delta function at the given point.

    Parameters:
    N (int): The number of grid points in each direction.
    radius (float): The radius of the circular domain.
    source (tuple): The coordinates of the source point.
    plot_M_mask (bool): Whether to plot the Laplacian matrix and the mask.

    Returns:
    np.ndarray: The solution to the steady state heat equation.
    """

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

    if plot_M_mask:
    # Plot the Laplacian matrix and the mask next to each other
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
    x = spsolve(M, b)
    return x

if __name__ == '__main__':
    N = 100
    radius = 2
    x = solve_steady_state(N, radius, plot_M_mask=True)
    plt.imshow(x.reshape((N, N)), cmap='viridis', norm=LogNorm())
    plt.colorbar(label="Value")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.title("Steady State Solution")
    plt.show()
