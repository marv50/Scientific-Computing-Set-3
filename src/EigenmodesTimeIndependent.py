import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
import matplotlib.pyplot as plt
import time


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
                 format='lil')
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
                 format='lil')
    return M, Nx, Ny, None


def laplacian_circular(N, radius=0.8, xsize=1, ysize=1):
    """
    Constructs the 2D Laplacian matrix for a circular domain.

    The domain is defined on a square grid of size N x N (with coordinates in [-1, 1])
    and only those points satisfying x^2 + y^2 <= radius^2 are kept.

    Returns:
        M: Sparse Laplacian matrix for the circular domain.
        Nx, Ny: The grid dimensions (both equal to N).
        mask: A 2D boolean array (shape N x N) where True indicates the point is inside the circle.
    """
    size = N * N
    # Create coordinates on a square in [-1, 1] x [-1, 1]
    x = np.linspace(-xsize, xsize, N)
    y = np.linspace(-ysize, ysize, N)
    X, Y = np.meshgrid(x, y)
    # Boolean mask for points inside the circle
    mask = (X**2 + Y**2) <= radius**2

    main_diag = -4 * np.ones(size)
    side_diag = np.ones(size - 1)
    # Remove wrap-around connections between rows
    side_diag[np.arange(1, size) % N == 0] = 0
    up_down_diag = np.ones(size - N)

    M = sp.diags([main_diag, side_diag, side_diag, up_down_diag, up_down_diag],
                 [0, -1, 1, -N, N],
                 shape=(size, size),
                 format='lil')
    # Zero out rows for points outside the circular domain
    for i, val in enumerate(mask.flatten()):
        if not val:
            M[i, :] = 0
            M[i, i] = 1

    return M, N, N, mask


def simulate_domain(domain, N, solver="sparse", k=6):
    """
    Simulates the eigenvalue problem for the Laplacian on a specified domain.

    Parameters:
        domain (str): One of "square", "rectangle", or "circle" (or "circular").
        N (int): Grid size parameter.
        solver (str): "sparse" or "dense" indicating which eigenvalue solver to use.
        k (int): Number of eigenmodes to compute.

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

    if solver == "sparse":
        # Use the sparse eigenvalue solver (spla.eigs)
        eigenvalues, eigenvectors = spla.eigs(M, k=k, which="SM")
        eigenvalues = eigenvalues.real  # take the real parts
        # Sort eigenpairs by the absolute value of the eigenvalues
        idx = np.argsort(np.abs(eigenvalues))
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

    elif solver == "dense":
        # Convert sparse matrix to dense array
        A = M.toarray()
        # Use dense solver for symmetric matrices: scipy.linalg.eigh()
        eigenvalues_all, eigenvectors_all = la.eig(A)
        # Sort eigenvalues by absolute value and select the k smallest
        idx = np.argsort(np.abs(eigenvalues_all))
        eigenvalues = eigenvalues_all[idx][:k]
        eigenvectors = eigenvectors_all[:, idx][:, :k]
    else:
        raise ValueError("Invalid solver type. Use 'sparse' or 'dense'.")

    results = {
        'M': M,
        'Nx': Nx,
        'Ny': Ny,
        'mask': mask,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors
    }
    return results


def compare_solver_performance(domain, N, num_runs):
    """
    Compares the performance of the sparse and dense eigenvalue solvers over multiple runs.

    Parameters:
        domain (str): Domain type ("square", "rectangle", or "circle").
        N (int): Grid size parameter.
        num_runs (int): Number of runs for timing each solver.

    This function times the eigenvalue decomposition using:
        - The sparse solver (spla.eigs): Efficient for large, sparse matrices.
        - The dense solver (scipy.linalg.eigh): Optimized for symmetric matrices.

    The function calculates and prints the mean and standard deviation of the run times.
    """
    print(
        f"\nComparing solvers for the {domain} domain with grid size parameter N = {N} over {num_runs} runs")

    sparse_times = []
    dense_times = []

    for run in range(num_runs):
        # Time the sparse solver
        start_time = time.perf_counter()
        results_sparse = simulate_domain(domain, N, solver="sparse")
        sparse_time = time.perf_counter() - start_time
        sparse_times.append(sparse_time)

        # Time the dense solver
        start_time = time.perf_counter()
        results_dense = simulate_domain(domain, N, solver="dense")
        dense_time = time.perf_counter() - start_time
        dense_times.append(dense_time)

    sparse_mean = np.mean(sparse_times)
    sparse_std = np.std(sparse_times)
    dense_mean = np.mean(dense_times)
    dense_std = np.std(dense_times)

    print(
        f"Sparse solver (spla.eigs) mean time: {sparse_mean:.6f} seconds, std: {sparse_std:.6f} seconds")
    print(
        f"Dense solver (scipy.linalg.eig) mean time: {dense_mean:.6f} seconds, std: {dense_std:.6f} seconds")
    print("\nEigenvalues from the last run of the sparse solver:")
    print(results_sparse['eigenvalues'])
    print("\nEigenvalues from the last run of the dense solver:")
    print(results_dense['eigenvalues'])

    stats = {
        "sparse_mean": sparse_mean,
        "sparse_std": sparse_std,
        "dense_mean": dense_mean,
        "dense_std": dense_std,
        "sparse_times": sparse_times,
        "dense_times": dense_times
    }

    return results_sparse, results_dense, stats


def plot_matrix(results, domain, save=False):
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
    if save:
        plt.savefig(f"fig/laplacian_{domain}.png")
    plt.show()


def plot_eigenmodes(results, domain, save=False):
    """
    Plots all available eigenmodes for the given domain as subplots in a single figure.
    Each subplot title now shows the eigenfrequency for that mode.

    Parameters:
        results: Dictionary returned from simulate_domain.
        domain (str): Domain name (for title display purposes).
    """
    Nx = results['Nx']
    Ny = results['Ny']
    mask = results['mask']
    eigenvectors = results['eigenvectors']
    eigenvalues = results['eigenvalues']
    num_modes = eigenvectors.shape[1]

    # Determine subplot grid size (roughly square layout)
    cols = min(num_modes, 3)
    rows = (num_modes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(num_modes):
        v = eigenvectors[:, i].real
        # Calculate eigenfrequency as sqrt(-eigenvalue) if the eigenvalue is negative
        eigenvalue = eigenvalues[i]
        freq = np.sqrt(-eigenvalue) if eigenvalue < 0 else np.nan

        if mask is None:
            field = v.reshape((Ny, Nx))
        else:
            field = np.zeros((Ny, Nx))
            valid_indices = np.where(mask.flatten())[0]
            temp = np.zeros(Nx * Ny)
            temp[valid_indices] = v[mask.flatten()]
            field = temp.reshape((Ny, Nx))

        ax = axes[i]
        im = ax.imshow(field, cmap="copper", extent=[0, Nx, 0, Ny])
        ax.set_title(f"Mode {i+1}\nFreq = {freq:.2f}", fontsize=18)
        fig.colorbar(im, ax=ax, shrink=0.7, aspect=10)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Eigenmodes for {domain.capitalize()} Domain", fontsize=22)
    plt.tight_layout()
    if save:
        plt.savefig(f"fig/eigenmodes_{domain}.png")
    plt.show()


def plot_performance_stats(performance_stats, domains, save=False):
    """
    Plots the mean execution times and standard deviations of the sparse and dense solvers
    for the specified domains.

    Parameters:
        performance_stats (dict): Dictionary where each key is a domain and the value is a
                                  dictionary of performance statistics.
        domains (list): List of domain names.
    """
    sparse_means = [performance_stats[d]["sparse_mean"] for d in domains]
    sparse_stds = [performance_stats[d]["sparse_std"] for d in domains]
    dense_means = [performance_stats[d]["dense_mean"] for d in domains]
    dense_stds = [performance_stats[d]["dense_std"] for d in domains]

    x = np.arange(len(domains))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x - width/2, sparse_means, width, yerr=sparse_stds,
           label='Sparse (eigs)', capsize=5)
    ax.bar(x + width/2, dense_means, width, yerr=dense_stds,
           label='Dense (eig)', capsize=5)

    ax.set_ylabel('Mean Execution Time (seconds)', fontsize=24)
    ax.set_title('Solver Performance Comparison by Domain', fontsize=28)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in domains], fontsize=22)
    ax.legend(fontsize=22)
    ax.tick_params(axis='both', labelsize=20)

    plt.tight_layout()
    if save:
        plt.savefig("fig/performance_comparison.png")
    plt.show()


def plot_domains_frequency_vs_N(domains, N_values, solver="sparse", k=6, save=False):
    """
    For each domain in the list, computes and plots the eigenfrequency vs. domain size (N)
    for multiple eigenmodes. The fundamental mode (mode 0) is plotted as a solid line, and 
    all higher modes are plotted as dashed lines. The legend is built with only two entries per
    domain: one for the fundamental and one for higher order modes.
    """
    plt.figure(figsize=(12, 9))
    # One color per domain
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(domains)))
    
    for idx, domain in enumerate(domains):
        # modes_freq will store frequencies for each mode across all N_values
        modes_freq = np.zeros((k, len(N_values)))
        
        # Loop over all grid sizes and compute frequencies once per domain
        for i, N in enumerate(N_values):
            results = simulate_domain(domain, N, solver=solver, k=k)
            eigenvalues = results['eigenvalues']
            for mode in range(k):
                # Only take the square-root if the eigenvalue is negative (as expected)
                eigenvalue = eigenvalues[mode]
                freq = np.sqrt(-eigenvalue) if eigenvalue < 0 else np.nan
                modes_freq[mode, i] = freq
        
        # Plot the modes for this domain
        for mode in range(k):
            if mode == 0:
                plt.plot(N_values, modes_freq[mode, :], marker='o', linestyle='-', color=colors[idx], alpha=0.7, label=f"{domain.capitalize()} Fundamental")
            else:
                plt.plot(N_values, modes_freq[mode, :], marker='o', linestyle='--', color=colors[idx], alpha=0.7)
        
        # Add dummy plot entries (invisible) to create clean legend entries for this domain
        plt.plot([], [], color=colors[idx], marker='o', linestyle='--', label=f"{domain.capitalize()} Higher order")
    
    plt.xlabel("Domain Size Parameter N", fontsize=24)
    plt.ylabel("Eigenfrequency", fontsize=24)
    plt.title("Eigenfrequencies vs Domain Size ", fontsize=28)
    plt.grid(True)
    plt.legend(fontsize=22)
    plt.tight_layout()
    if save:
        plt.savefig("fig/frequency_vs_N.png")
    plt.show()



if __name__ == "__main__":
    N = 30       # Grid size parameter (adjust as needed)
    num_runs = 10  # Number of runs for performance timing

    # Define the domains to test
    domains = ["square", "rectangle", "circle"]

    # Example: Plot eigenfrequency vs domain size for the square domain
    N_values = np.linspace(20, 60, 20, dtype=int)

    # Plot N vs frequency for all domains and for multiple eigenmodes (pooled per domain)
    plot_domains_frequency_vs_N(domains, N_values, solver="sparse", k=6, save=True)