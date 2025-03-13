import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


def laplacian_square(N):
    """
    Constructs the 2D Laplacian matrix for a square domain of grid size N x N.

    Returns:
        M: Sparse Laplacian matrix.
        Nx, Ny: Grid dimensions.
        mask: None (no mask is needed for a full square).
    """
    size = N * N
    main_diag = -4 * np.ones(size)
    side_diag = np.ones(size - 1)
    # Remove wrap-around connections between rows
    side_diag[np.arange(1, size) % N == 0] = 0
    up_down_diag = np.ones(size - N)

    M = sp.diags([main_diag, side_diag, side_diag, up_down_diag, up_down_diag],
                 [0, -1, 1, -N, N],
                 shape=(size, size),
                 format='csr')
    return M, N, N, None


def laplacian_rectangular(N):
    """
    Constructs the 2D Laplacian matrix for a rectangular domain with grid size (2N x N).

    Returns:
        M: Sparse Laplacian matrix.
        Nx, Ny: Grid dimensions (Nx = 2N, Ny = N).
        mask: None (no mask is needed for a full rectangle).
    """
    Nx = 2 * N
    Ny = N
    size = Nx * Ny
    main_diag = -4 * np.ones(size)
    side_diag = np.ones(size - 1)
    # Remove wrap-around connections (every Nx points)
    side_diag[np.arange(1, size) % Nx == 0] = 0
    up_down_diag = np.ones(size - Nx)

    M = sp.diags([main_diag, side_diag, side_diag, up_down_diag, up_down_diag],
                 [0, -1, 1, -Nx, Nx],
                 shape=(size, size),
                 format='csr')
    return M, Nx, Ny, None


def laplacian_circular(N, radius=0.8):
    """
    Constructs the 2D Laplacian matrix for a circular domain.

    The domain is defined on a square grid of size N x N (with coordinates in [-1, 1])
    and only those points satisfying x^2 + y^2 <= radius^2 are kept.

    Returns:
        M_circular: Sparse Laplacian matrix for the circular domain.
        Nx, Ny: The grid dimensions (both equal to N).
        mask: A 2D boolean array (shape N x N) where True indicates the point is inside the circle.
    """
    size = N * N
    # Create coordinates on a square in [-1, 1] x [-1, 1]
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    # Boolean mask for points inside the circle
    mask = (X**2 + Y**2) <= radius**2
    valid_indices = np.where(mask.flatten())[0]

    main_diag = -4 * np.ones(size)
    side_diag = np.ones(size - 1)
    side_diag[np.arange(1, size) % N == 0] = 0
    up_down_diag = np.ones(size - N)

    M_full = sp.diags([main_diag, side_diag, side_diag, up_down_diag, up_down_diag],
                      [0, -1, 1, -N, N],
                      shape=(size, size),
                      format='csr')
    # Restrict to points inside the circle
    M_circular = M_full.tocsr()[valid_indices, :][:, valid_indices]
    return M_circular, N, N, mask


def simulate_domain(domain, N):
    """
    Simulates the eigenvalue problem for the Laplacian on a specified domain.

    Parameters:
        domain (str): One of "square", "rectangle", or "circle" (or "circular").
        N (int): Grid size parameter.

    Returns:
        results: A dictionary containing the Laplacian matrix M, grid dimensions (Nx, Ny),
                 mask (if applicable), eigenvalues, and eigenvectors.
    """
    domain = domain.lower()
    if domain == "square":
        M, Nx, Ny, mask = laplacian_square(N)
    elif domain == "rectangle":
        M, Nx, Ny, mask = laplacian_rectangular(N)
    elif domain in ["circle", "circular"]:
        M, Nx, Ny, mask = laplacian_circular(N)
    else:
        raise ValueError(
            "Invalid domain. Choose 'square', 'rectangle', or 'circle'.")

    # Compute the 5 smallest eigenvalues and corresponding eigenvectors.
    eigenvalues, eigenvectors = spla.eigs(M, k=5, which="SM")

    results = {
        'M': M,
        'Nx': Nx,
        'Ny': Ny,
        'mask': mask,
        'eigenvalues': eigenvalues.real,
        'eigenvectors': eigenvectors
    }
    return results


def plot_matrix(results, domain):
    """
    Plots the dense representation of the Laplacian matrix.

    Parameters:
        results: Dictionary returned from simulate_domain.
        domain (str): Domain name (for title display purposes).
    """
    M = results['M']
    plt.figure(figsize=(6, 5))
    plt.imshow(M.toarray(), cmap="viridis")
    plt.colorbar(label="Matrix Value")
    plt.title(
        f"Visualization of Laplacian Matrix ({domain.capitalize()} Domain)")
    plt.show()


def plot_eigenmode(results, domain):
    """
    Plots the first eigenmode on the domain.

    Parameters:
        results: Dictionary returned from simulate_domain.
        domain (str): Domain name (for title display purposes).
    """
    Nx = results['Nx']
    Ny = results['Ny']
    mask = results['mask']
    eigenvalues = results['eigenvalues']
    eigenvectors = results['eigenvectors']

    # Extract first eigenvector
    v1 = eigenvectors[:, 0].real
    if mask is None:
        # For square or rectangular domains, reshape directly.
        field = v1.reshape((Ny, Nx))
    else:
        # For circular domain, create a full grid and insert valid values.
        field = np.zeros((Ny, Nx))
        valid_indices = np.where(mask.flatten())[0]
        temp = np.zeros(Nx * Ny)
        temp[valid_indices] = v1
        field = temp.reshape((Ny, Nx))

    plt.figure(figsize=(6, 5))
    plt.imshow(field, cmap="copper", extent=[0, Nx, 0, Ny])
    plt.colorbar(label="Amplitude")
    plt.title(f"First Eigenmode on {domain.capitalize()} Domain")
    plt.show()


if __name__ == "__main__":

    N = 50
    domains = ["square", "rectangle", "circle"]

    for domain in domains:
        results = simulate_domain(domain, N)
        print(f"Eigenvalues for {domain.capitalize()} domain:")
        print(results['eigenvalues'])
        plot_matrix(results, domain)
        plot_eigenmode(results, domain)
