# Scientific-Computing-Set-3


## Direct methods for solving steady state problems

### Files
- `src/DirectSteadyState.py`: Contain the implementation of the direct solver.

### Description

Solves the steady state heat equation for a circular domain with a source at a given point.
The source is a point source, i.e. the source is a delta function at the given point.
If the file is ran a solution will be plotted.

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

## Notebook

Run notebook.ipynb in order to have all relevant, output generating functions in one file.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Numba

All dependencies can be installed via `pip`.

```
pip install requirements.txt
```