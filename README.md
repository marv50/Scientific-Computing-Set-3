# Scientific-Computing-Set-3

## Leapfrog Integration for a Simple and Driven Harmonic Oscillator

This repository contains a Python script that implements the Leapfrog integration method to simulate the motion of a harmonic oscillator using Hooke's Law. The script also includes a comparison with the RK45 integrator to analyze energy conservation properties. Additionally, a driven oscillator variant is implemented to study resonance effects.

### Files

- `src/leapfrog.py`: Contains the implementation of the Leapfrog integration method for both simple and driven harmonic oscillators, as well as a comparison with the RK45 integrator.

### Description

The script performs the following tasks:

1. **Leapfrog Integration for Simple Harmonic Oscillator**:
   - Implements the Leapfrog method to integrate the equations of motion for a harmonic oscillator.
   - Compares the energy conservation properties with the RK45 integrator.

2. **Leapfrog Integration for Driven Harmonic Oscillator**:
   - Extends the Leapfrog method to include an external time-dependent sinusoidal driving force.
   - Analyzes the resonance effects by plotting the phase space for various driving frequencies.

### Plots

The script generates the following plots:
- Position vs Time
- Velocity vs Time
- Energy Conservation (Leapfrog vs RK45)
- Phase Space for Driven Oscillator

The plots are saved as `fig/leapfrog.png`.