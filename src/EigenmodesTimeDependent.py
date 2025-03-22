import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from EigenmodesTimeIndependent import *

def T(t, lam, c=1, A=1, B=1):
    """Time-dependent component of the wave equation solution."""
    return A * np.cos(c * lam * t) + B * np.sin(c * lam * t)

def simulate_time_dependent(N, time_steps, domain, mode_index=0):
    """Simulate time-dependent wave equation for the given domain."""
    results = simulate_domain(domain, N)
    eigenvectors = results['eigenvectors']
    eigenvalues = results['eigenvalues']
    eigenvalue = eigenvalues[mode_index]
    v = eigenvectors.real[:, mode_index]  
    mask = results["mask"]
    lam = np.sqrt(-eigenvalue) if eigenvalue < 0 else 0.1
    
    # Get the correct dimensions from the results
    Nx = results['Nx']
    Ny = results['Ny']
    
    # Initialize results array with the correct dimensions
    results_array = np.zeros((time_steps, Ny, Nx))

    for i in range(time_steps):
        if mask is None:
            field = v.reshape((Ny, Nx))
        else:
            field = np.zeros((Ny, Nx))
            valid_indices = np.where(mask.flatten())[0]
            temp = np.zeros(Nx * Ny)
            temp[valid_indices] = v[valid_indices]
            field = temp.reshape((Ny, Nx))
        
        results_array[i] = field * T(i, lam)
    
    return results_array, lam

def animate_time_dependent(N, time_steps, domain, mode_index=0, save_animation=False):
    """Create enhanced animation of the time-dependent solution."""
    results, lam = simulate_time_dependent(N, time_steps, domain, mode_index)
    
    # Find global min/max for consistent color scaling
    vmin, vmax = np.min(results), np.max(results)
    abs_max = max(abs(vmin), abs(vmax))
    
    # Create symmetric color scale around zero
    norm = Normalize(vmin=-abs_max, vmax=abs_max)
    
    # Create figure with two subplots - 2D heatmap and 3D surface
    fig = plt.figure(figsize=(12, 6))
    
    # 2D heatmap subplot
    ax1 = fig.add_subplot(121)
    im = ax1.imshow(results[0], cmap='seismic', norm=norm, 
                   interpolation='bilinear', animated=True)
    ax1.set_title("2D Wave Pattern")
    fig.colorbar(im, ax=ax1, label="Amplitude")
    
    # 3D surface subplot
    ax2 = fig.add_subplot(122, projection='3d')
    Ny, Nx = results[0].shape
    X, Y = np.meshgrid(range(Nx), range(Ny))
    
    # Create the initial surface plot
    surf = ax2.plot_surface(X, Y, results[0], cmap='viridis', 
                          norm=norm, edgecolor='none', rstride=2, cstride=2)
    ax2.set_title("3D Wave Surface")
    ax2.set_zlim(-abs_max, abs_max)
    
    # Add a timer display
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, color='white', 
                        bbox=dict(facecolor='black', alpha=0.5))
    
    # Animation update function
    def update(frame):
        # Update 2D image
        im.set_array(results[frame])
        
        # Update 3D surface (remove old and create new for better performance)
        ax2.clear()
        surf = ax2.plot_surface(X, Y, results[frame], cmap='viridis', 
                              norm=norm, edgecolor='none', rstride=2, cstride=2)
        ax2.set_zlim(-abs_max, abs_max)
        ax2.set_title("3D Wave Surface")
        
        # Update timer
        time_text.set_text(f'Time: {frame/time_steps:.2f}T (T=2π/{lam:.2f})')
        
        return im, surf, time_text
    
    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=time_steps, interval=30, blit=False)
    
    # Add a main title
    plt.suptitle(f"Wave Equation on {domain.capitalize()} Domain - Mode {mode_index+1}", fontsize=16)
    plt.tight_layout()
    
    # Save animation if requested
    if save_animation:
        ani.save(f'wave_{domain}_mode{mode_index+1}.mp4', writer='ffmpeg', fps=30)
    
    plt.show()
    return ani

def plot_time_series(N, time_steps, domain, mode_index=0, num_frames=6, save_plot=False):
    """
    Create a static plot showing the development of the wave over time.
    Displays multiple frames from the time evolution as a grid of subplots.
    """
    results, lam = simulate_time_dependent(N, time_steps, domain, mode_index)
    
    # Find global min/max for consistent color scaling
    vmin, vmax = np.min(results), np.max(results)
    abs_max = max(abs(vmin), abs(vmax))
    
    # Create symmetric color scale around zero
    norm = Normalize(vmin=-abs_max, vmax=abs_max)
    
    # Calculate the grid dimensions for the subplots
    grid_size = int(np.ceil(np.sqrt(num_frames)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 10))
    
    # Ensure axes is a 2D array for consistent indexing
    if grid_size == 1:
        axes = np.array([[axes]])
    elif grid_size > 1 and num_frames <= grid_size:
        axes = axes.reshape(1, -1)
    
    # Select frames evenly distributed throughout the time series
    frame_indices = np.linspace(0, time_steps-1, num_frames, dtype=int)
    
    # Create a list to store image references for a shared colorbar
    images = []
    
    # Plot each selected frame
    frame_count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if frame_count < num_frames:
                frame_idx = frame_indices[frame_count]
                im = axes[i, j].imshow(results[frame_idx], cmap='seismic', 
                                     norm=norm, interpolation='bilinear')
                axes[i, j].set_title(f'Time: {frame_idx/time_steps:.2f}T')
                images.append(im)
                frame_count += 1
            else:
                # Hide unused subplots
                axes[i, j].axis('off')
    
    # Add a colorbar that applies to all subplots
    fig.colorbar(images[0], ax=axes.ravel().tolist(), label="Amplitude", shrink=0.7)
    
    # Add a main title
    plt.suptitle(f"Wave Evolution on {domain.capitalize()} Domain - Mode {mode_index+1}\n(T=2π/{lam:.2f})", 
                fontsize=16)
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        plt.savefig(f'wave_timeseries_{domain}_mode{mode_index+1}.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_3d_time_series(N, time_steps, domain, mode_index=0, num_frames=4, save_plot=False):
    """
    Create a static 3D plot showing the development of the wave over time.
    Displays multiple 3D surface plots from the time evolution as a grid.
    """
    results, lam = simulate_time_dependent(N, time_steps, domain, mode_index)
    
    # Find global min/max for consistent color scaling
    vmin, vmax = np.min(results), np.max(results)
    abs_max = max(abs(vmin), abs(vmax))
    
    # Create symmetric color scale around zero
    norm = Normalize(vmin=-abs_max, vmax=abs_max)
    
    # Setup grid for 3D plots
    grid_size = int(np.ceil(np.sqrt(num_frames)))
    fig = plt.figure(figsize=(15, 12))
    
    # Select frames evenly distributed throughout the time series
    frame_indices = np.linspace(0, time_steps-1, num_frames, dtype=int)
    
    # Create mesh grid for 3D plotting
    Ny, Nx = results[0].shape
    X, Y = np.meshgrid(range(Nx), range(Ny))
    
    # Plot each selected frame as 3D surface
    for i, frame_idx in enumerate(frame_indices):
        if i < num_frames:
            ax = fig.add_subplot(grid_size, grid_size, i+1, projection='3d')
            surf = ax.plot_surface(X, Y, results[frame_idx], cmap='viridis', 
                                  norm=norm, edgecolor='none', rstride=2, cstride=2)
            ax.set_zlim(-abs_max, abs_max)
            ax.set_title(f'Time: {frame_idx/time_steps:.2f}T')
            
            # Adjust view angle for better visualization
            ax.view_init(elev=30, azim=45)
    
    # Add a main title
    plt.suptitle(f"3D Wave Evolution on {domain.capitalize()} Domain - Mode {mode_index+1}\n(T=2π/{lam:.2f})", 
                fontsize=16)
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        plt.savefig(f'wave_3d_timeseries_{domain}_mode{mode_index+1}.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def compare_modes(N, time_steps, domain, num_modes=4, layout=None):
    """
    Create a multi-panel animation comparing different eigenmodes.
    
    Parameters:
    -----------
    N : int
        Grid resolution
    time_steps : int
        Number of time steps to simulate
    domain : str
        Domain shape ('square', 'rectangle', 'circle', etc.)
    num_modes : int
        Number of modes to compare
    layout : tuple, optional
        Layout of the subplots as (rows, cols). If None, automatically determined.
    """
    # Determine subplot layout if not specified
    if layout is None:
        # Calculate an appropriate layout based on num_modes
        rows = int(np.ceil(np.sqrt(num_modes)))
        cols = int(np.ceil(num_modes / rows))
        layout = (rows, cols)
    else:
        rows, cols = layout
        # Ensure layout has enough panels
        if rows * cols < num_modes:
            rows = int(np.ceil(np.sqrt(num_modes)))
            cols = int(np.ceil(num_modes / rows))
            layout = (rows, cols)
            print(f"Warning: Specified layout is too small. Using {layout} instead.")
    
    # Create figure with subplots for each mode
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    
    # Ensure axes is always a 2D array for consistent indexing
    if num_modes == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    animations = []
    all_results = []
    all_lams = []
    
    # Calculate max value across all modes for consistent color scaling
    max_val = 0
    for mode in range(num_modes):
        results, lam = simulate_time_dependent(N, time_steps, domain, mode)
        all_results.append(results)
        all_lams.append(lam)
        max_val = max(max_val, np.max(np.abs(results)))
    
    # Create a symmetric color scale around zero
    norm = Normalize(vmin=-max_val, vmax=max_val)
    
    # Set up each subplot
    for i in range(rows):
        for j in range(cols):
            mode_idx = i * cols + j
            if mode_idx < num_modes:
                im = axes[i, j].imshow(all_results[mode_idx][0], cmap='seismic', norm=norm, 
                                      interpolation='bilinear', animated=True)
                axes[i, j].set_title(f"Mode {mode_idx+1} (f={all_lams[mode_idx]:.2f})")
                animations.append(im)
            else:
                # Hide unused subplots
                axes[i, j].axis('off')
    
    # Add a colorbar
    fig.colorbar(animations[0], ax=axes.ravel().tolist(), label="Amplitude", shrink=0.7)
    
    # Animation update function
    def update(frame):
        for i in range(num_modes):
            animations[i].set_array(all_results[i][frame])
        return animations
    
    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=time_steps, interval=40, blit=False)
    
    plt.suptitle(f"Wave Equation Modes on {domain.capitalize()} Domain", fontsize=16)
    plt.tight_layout()
    plt.show()
    return ani

def compare_modes_static(N, time_steps, domain, num_modes=4, time_point=None, layout=None, save_plot=False):
    """
    Create a static comparison of different eigenmodes at a specific time point.
    
    Parameters:
    -----------
    N : int
        Grid resolution
    time_steps : int
        Number of time steps to simulate
    domain : str
        Domain shape ('square', 'rectangle', 'circle', etc.)
    num_modes : int
        Number of modes to compare
    time_point : int, optional
        Time step to display. If None, uses the time step with maximum amplitude.
    layout : tuple, optional
        Layout of the subplots as (rows, cols). If None, automatically determined.
    save_plot : bool
        Whether to save the plot to a file
    """
    # Determine subplot layout if not specified
    if layout is None:
        # Calculate an appropriate layout based on num_modes
        rows = int(np.ceil(np.sqrt(num_modes)))
        cols = int(np.ceil(num_modes / rows))
        layout = (rows, cols)
    else:
        rows, cols = layout
        # Ensure layout has enough panels
        if rows * cols < num_modes:
            rows = int(np.ceil(np.sqrt(num_modes)))
            cols = int(np.ceil(num_modes / rows))
            layout = (rows, cols)
            print(f"Warning: Specified layout is too small. Using {layout} instead.")
    
    # Create figure with subplots for each mode
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    
    # Ensure axes is always a 2D array for consistent indexing
    if num_modes == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    all_results = []
    all_lams = []
    
    # Simulate all modes
    for mode in range(num_modes):
        results, lam = simulate_time_dependent(N, time_steps, domain, mode)
        all_results.append(results)
        all_lams.append(lam)
    
    # Find global max value for consistent color scaling
    max_val = max([np.max(np.abs(res)) for res in all_results])
    norm = Normalize(vmin=-max_val, vmax=max_val)
    
    # Determine time point to display
    if time_point is None:
        # Find time point with maximum amplitude across all modes
        max_amplitudes = [np.max(np.abs(res)) for res in all_results]
        max_mode = np.argmax(max_amplitudes)
        max_frame = np.argmax(np.max(np.abs(all_results[max_mode]), axis=(1, 2)))
        time_point = max_frame
    
    # Plot each mode at the selected time point
    for i in range(rows):
        for j in range(cols):
            mode_idx = i * cols + j
            if mode_idx < num_modes:
                im = axes[i, j].imshow(all_results[mode_idx][time_point], cmap='seismic', 
                                      norm=norm, interpolation='bilinear')
                axes[i, j].set_title(f"Mode {mode_idx+1} (f={all_lams[mode_idx]:.2f})")
            else:
                # Hide unused subplots
                axes[i, j].axis('off')
    
    # Add a colorbar
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Amplitude", shrink=0.7)
    
    plt.suptitle(f"Wave Equation Modes on {domain.capitalize()} Domain\nTime: {time_point/time_steps:.2f}T", 
                fontsize=16)
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        plt.savefig(f'wave_modes_comparison_{domain}_{num_modes}modes.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

if __name__ == "__main__":
    N = 70  # Grid size
    time_steps = 100  # Number of frames in the animation
    
    # Example usage of the enhanced functions
    
    # 1. Visualization of a single mode with animation
    # animate_time_dependent(N, time_steps, 'circle', mode_index=2)
    
    # 2. Static plot of time evolution for a single mode
    # plot_time_series(N, time_steps, 'square', mode_index=0, num_frames=9)
    
    # 3. Static 3D plot of time evolution
    # plot_3d_time_series(N, time_steps, 'rectangle', mode_index=1, num_frames=4)
    
    # 4. Compare different modes with animation
    # compare_modes(N, time_steps, 'square', num_modes=6, layout=(2, 3))
    
    # 5. Static comparison of different modes
    # compare_modes_static(N, time_steps, 'circle', num_modes=9, layout=(3, 3))
    
    # Example for running multiple domains
    for domain in ['square', 'rectangle', 'circle']:
        print(f"Comparing modes for {domain} domain...")
        # You can now easily adjust the number of modes and layout
        compare_modes(N, time_steps, domain, num_modes=6, layout=(2, 2))
        
        # Create static visualizations
        plot_time_series(N, time_steps, domain, mode_index=0, num_frames=6)