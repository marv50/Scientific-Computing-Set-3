"""
Leapfrog Integration for a Simple and Driven Harmonic Oscillator

This script implements the Leapfrog integration method to simulate the motion of a
harmonic oscillator using Hooke's Law. It also includes a comparison with the RK45
integrator to analyze energy conservation properties. Additionally, a driven oscillator
variant is implemented to study resonance effects.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def hooke_force(x, k):
    """
    Computes the restoring force of a simple harmonic oscillator using Hooke's Law.

    Parameters:
    x (float): Displacement from equilibrium position.
    k (float): Spring constant.

    Returns:
    float: Restoring force (-k * x).
    """
    return -k * x

def leapfrog(x, v, t, v_half, k, m, dt, steps):
    """
    Implements the Leapfrog integration method for a harmonic oscillator.

    Parameters:
    x (float): Initial position.
    v (float): Initial velocity.
    t (float): Initial time.
    v_half (float): Half-step velocity initialization.
    k (float): Spring constant.
    m (float): Mass of the oscillator.
    dt (float): Time step size.
    steps (int): Number of time steps to integrate.

    Returns:
    tuple: Arrays containing time, position, velocity, and total energy values.
    """
    t_vals, x_vals, v_vals, energy_vals = [t], [x], [v], [0.5 * m * v**2 + 0.5 * k * x**2]

    for _ in range(steps):
        v_half += 0.5 * dt * (hooke_force(x, k) / m) # Update velocity at half step
        x += dt * v_half # Update position
        v_half += 0.5 * dt * (hooke_force(x, k) / m) # Complete velocity update
        t += dt # Time update

        # Store values
        v = v_half + 0.5 * dt * (hooke_force(x, k) / m) # Approximate full step velocity for plotting
        energy = 0.5 * m * v**2 + 0.5 * k * x**2 # Total energy
        t_vals.append(t), x_vals.append(x), v_vals.append(v), energy_vals.append(energy)

    return t_vals, x_vals, v_vals, energy_vals

def rk45_oscillator(t, y, k, m):
    """
    Defines the system of ODEs for a simple harmonic oscillator to be solved using RK45.

    Parameters:
    t (float): Time variable.
    y (list): State vector [x, v], where x is position and v is velocity.
    k (float): Spring constant.
    m (float): Mass of the oscillator.

    Returns:
    list: First-order derivatives [dx/dt, dv/dt].
    """
    x, v = y
    return [v, hooke_force(x, k) / m]

def leapfrog_driven(omega_d, k, A, m, dt, steps):
    """
    Implements the Leapfrog method for a driven harmonic oscillator with an external sinusoidal force.

    Parameters:
    omega_d (float): Driving frequency of the external force.
    k (float): Spring constant.
    A (float): Amplitude of the driving force.
    m (float): Mass of the oscillator.
    dt (float): Time step size.
    steps (int): Number of time steps to integrate.

    Returns:
    tuple: Arrays containing position and velocity values.
    """
    x, v = 1.0, 0.0
    v_half = v - 0.5 * dt * (hooke_force(x, k) / m + A * np.sin(omega_d * 0) / m)
    x_vals, v_vals = [], []

    for i in range(steps):
        t = i * dt
        v_half += 0.5 * dt * (hooke_force(x, k) / m + A * np.sin(omega_d * t) / m)
        x += dt * v_half
        v_half += 0.5 * dt * (hooke_force(x, k) / m + A * np.sin(omega_d * t) / m)

        v = v_half + 0.5 * dt * (hooke_force(x, k) / m + A * np.sin(omega_d * t) / m)
        x_vals.append(x)
        v_vals.append(v)

    return x_vals, v_vals

if __name__ == '__main__':
    # Parameters
    k_values = [0.5, 1.0, 2.0] # Different spring constants for comparison
    m = 1.0 # Mass
    omega_d = 1.0 # Driving frequency for J
    A = 0.5 # Driving amplitude for J
    dt = 0.1 # Time step
    t_max = 50 # Total time
    steps = int(t_max / dt)

    # Initial conditions
    x0 = 1.0 # Initial position
    v0 = 0.0 # Initial velocity
    t0 = 0.0 # Initial time

    # Plotting Leapfrog results for different k values
    fig, axs = plt.subplots(4, 1, figsize=(8, 12))

    for i, k in enumerate(k_values):
        v_half = v0 - 0.5 * dt * (hooke_force(x0, k) / m) # Half-step velocity initialization
        t_vals, x_vals, v_vals, energy_vals = leapfrog(x0, v0, t0, v_half, k, m, dt, steps)

        # Position and velocity in separate subplots
        axs[0].plot(t_vals, x_vals, label=f"Position (k={k})")
        axs[1].plot(t_vals, v_vals, linestyle="dashed", label=f"Velocity (k={k})")

        # Energy conservation with RK45
        sol = solve_ivp(rk45_oscillator, [0, t_max], [x0, v0], args=(k, m), t_eval=np.linspace(0, t_max, steps))
        energy_rk45 = 0.5 * m * sol.y[1]**2 + 0.5 * k * sol.y[0]**2
        axs[2].plot(t_vals, energy_vals, label=f"Leapfrog Energy (k={k})")
        axs[2].plot(sol.t, energy_rk45, linestyle="dashed", label=f"RK45 Energy (k={k})")

    # Set titles and legends for better readability
    axs[0].set_title("Position vs Time")
    axs[0].legend()

    axs[1].set_title("Velocity vs Time")
    axs[1].legend()

    axs[2].set_title("Energy Conservation")
    axs[2].legend()

    # Phase plots for different driving frequencies
    for omega in [0.8, 1.0, 1.2]: # Resonant and non-resonant cases
        x_vals, v_vals = leapfrog_driven(omega, k_values[1], A, m, dt, steps)
        axs[3].plot(x_vals, v_vals, label=f"ω={omega}")

    axs[3].set_title("Phase Space for Driven Oscillator")
    axs[3].legend()

    plt.tight_layout()
    plt.savefig('fig/leapfrog.png')
