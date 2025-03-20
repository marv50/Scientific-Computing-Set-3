import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from EigenmodesTimeIndependent import *

def T(t, lam, c=1, A=1, B=1):
    return A * np.cos(c * lam * t) + B * np.sin(c * lam * t)

def simulate_time_dependent(N, time_steps, domain):
    results_array = np.zeros((time_steps, N, N))

    results = simulate_domain(domain, N)
    eigenvectors = results['eigenvectors']
    eigenvalues = results['eigenvalues']
    first_eigenvalue = eigenvalues[0]
    v = eigenvectors.real[:, 0]  
    mask = results["mask"]
    lam = np.sqrt(-first_eigenvalue)

    Ny, Nx = N, N

    for i in range(time_steps):
        if mask is None:
            field = v.reshape((Ny, Nx))
        else:
            field = np.zeros((Ny, Nx))
            valid_indices = np.where(mask.flatten())[0]
            temp = np.zeros(Nx * Ny)
            temp[valid_indices] = v[mask.flatten()]
            field = temp.reshape((Ny, Nx))
        
        results_array[i] = field * T(i, lam)
    return results_array

def animate_time_dependent(N, time_steps, domain):
    results = simulate_time_dependent(N, time_steps, domain)
    fig, ax = plt.subplots()
    
    # Initialize with first frame
    im = ax.imshow(results[0], cmap='turbo', interpolation='nearest', animated=True)
    
    def animate(i):
        im.set_array(results[i])
        return [im]
    
    ani = animation.FuncAnimation(fig, animate, frames=time_steps, interval=10, blit=False)

    plt.show()

if __name__ == "__main__":
    N = 100
    time_steps = 300  # Increase time steps for better animation
    animate_time_dependent(N, time_steps, "square")
