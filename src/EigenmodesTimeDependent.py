import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import animation

# Import the simulation function
if __name__ == '__main__':
    from EigenmodesTimeIndependent import simulate_domain
else:
    from src.EigenmodesTimeIndependent import simulate_domain

def T(t, lam, c=1, A=1, B=1):
    """Time-dependent component."""
    return A * np.cos(c * lam * t) + B * np.sin(c * lam * t)

def simulate_time_dependent(N, time_steps, domain, mode_index):
    """Compute time evolution for one eigenmode."""
    results = simulate_domain(domain, N)
    eigenvectors = results['eigenvectors']
    eigenvalues = results['eigenvalues']
    eigenvalue = eigenvalues[mode_index]
    v = eigenvectors.real[:, mode_index]
    lam = np.sqrt(-eigenvalue) if eigenvalue < 0 else 0.1  # avoid zero frequency
    Nx, Ny = results['Nx'], results['Ny']
    
    # Prepare the spatial field for this mode
    field0 = v.reshape((Ny, Nx))
    
    # Create time evolution frames
    frames = [field0 * T(t, lam) for t in range(time_steps)]
    return np.array(frames), lam

def plot_mode_evolution(N, time_steps, domain, mode_indices=[0, 1, 2, 3], num_frames=5, save_path=None):
    """
    Create a static plot showing the time evolution of multiple eigenmodes.
    
    Each row corresponds to a different mode, and each column shows a snapshot in time.
    """
    all_frames = []
    lam_list = []
    for m in mode_indices:
        frames, lam = simulate_time_dependent(N, time_steps, domain, m)
        all_frames.append(frames)
        lam_list.append(lam)
    
    # Choose evenly spaced time indices
    time_idxs = np.linspace(0, time_steps - 1, num_frames, dtype=int)
    
    # Determine global amplitude range for consistent color scaling
    global_max = max(np.max(np.abs(frames)) for frames in all_frames)
    norm = Normalize(vmin=-global_max, vmax=global_max)
    
    num_modes = len(mode_indices)
    fig, axes = plt.subplots(num_modes, num_frames, figsize=(3*num_frames, 3*num_modes))
    
    # If there's only one row or column, ensure axes is 2D
    if num_modes == 1:
        axes = axes[np.newaxis, :]
    if num_frames == 1:
        axes = axes[:, np.newaxis]
    
    for i, frames in enumerate(all_frames):
        for j, t_idx in enumerate(time_idxs):
            ax = axes[i, j]
            im = ax.imshow(frames[t_idx], cmap='seismic', norm=norm, interpolation='none')
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"Mode {mode_indices[i]+1}\nf={lam_list[i]:.2f}", fontsize=18)
            ax.set_title(f"t = {t_idx/time_steps:.2f}", fontsize=20)
    
    # Add a single colorbar for all subplots with increased label font size
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', label='Amplitude')
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Amplitude', fontsize=20)

    
    fig.suptitle(f"Wave Eigenmode Evolution on {domain.capitalize()} Domain", fontsize=24)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def create_multiple_mode_animation(N, time_steps, domain, mode_indices=[0, 1, 2], fps=10, save_path=None):
    """Create an animation showing the time evolution of multiple eigenmodes."""
    num_modes = len(mode_indices)
    frames_list = []
    lam_list = []
    
    # Simulate time evolution for each mode
    for mode_index in mode_indices:
        frames, lam = simulate_time_dependent(N, time_steps, domain, mode_index)
        frames_list.append(frames)
        lam_list.append(lam)

    # Set up the figure and axes for subplots
    fig, axes = plt.subplots(1, num_modes, figsize=(5*num_modes, 5))
    norm = Normalize(vmin=-np.max(np.abs(frames_list)), vmax=np.max(np.abs(frames_list)))

    # Create initial plot for each mode
    ims = []
    for i, frames in enumerate(frames_list):
        im = axes[i].imshow(frames[0], cmap='seismic', norm=norm, interpolation='none')
        axes[i].set_title(f"Mode {mode_indices[i]+1} (f={lam_list[i]:.2f})", fontsize=12)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        ims.append(im)

    # Update function for animation
    def update(frame_idx):
        for i, frames in enumerate(frames_list):
            ims[i].set_array(frames[frame_idx])
        return ims

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=time_steps, interval=1000/fps, blit=False)
    
    # Add a colorbar to the figure
    cbar = fig.colorbar(ims[0], ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Amplitude')
    
    if save_path:
        ani.save(save_path, writer='pillow', fps=fps)
    return ani, fig

if __name__ == "__main__":
    N = 70           # grid resolution
    time_steps = 100 # number of time frames
    domain = 'circle'  # choose from 'square', 'rectangle', 'circle', etc.
    
    # Static multi-mode evolution plot
    fig_static = plot_mode_evolution(
        N, time_steps, domain, mode_indices=[0, 2, 4],
        num_frames=5)
    

