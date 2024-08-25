"""
my animation creation was running unbelievably slow even though the vectorized
Euler's method finished running immediately. 

I found this repo solving the same problem
https://github.com/Svaught598/Computational-Physics/blob/master/PythonProgramming/Homework%2007/Newman_9.5.py

the solution seems to use a generator function to give the animator the data
as it comes in... which I don't really understand since I was able to generate
all of the data within a second using my vectorized function...

I was wasting too much time on this, so I'm copying their method without asking anymore questions :)
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Generator
import matplotlib.animation as animation

# constants
L = 1.0                       # length of string in m
d = 0.1                       # mean of gaussian pulse in m
nu = 100                      # wave speed in m/s
C = 1.0                       # amplitude of velocity wave in m/s
sigma = 0.3                   # standard deviation of Gaussian pulse

# simulation parameters
h = 5.0005e-5                 # time step size in s
total_time = 100.0e-3         # time to run to in s
num_steps = int(total_time/h) # number of time steps
N = 200                       # number of grid divisions
a = L/N                       # grid spacing in m

def psi(x: float, d:float, sigma:float, C:float, L:float) -> float:
    """Wave displacement velocity profile at t = 0"""
    return C * x*(L - x)/L**2 * np.exp(-(x-d)**2 / (2*sigma**2))

def euler(phi_grid: np.ndarray, psi_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Euler's method for returning the entire displacement and velocity history"""
    # get the number of grid points
    N = phi_grid.shape[1] - 1

    for t in range(num_steps-1):
        # only update the interior points -- the boundary points are fixed
        phi_grid[t+1, 1:N] = phi_grid[t, 1:N] + h * psi_grid[t, 1:N]
        psi_grid[t+1, 1:N] = psi_grid[t, 1:N] + h * (nu/a)**2 * (phi_grid[t, 2:N+1] + phi_grid[t, 0:N-1] - 2*phi_grid[t, 1:N]) 
    return phi_grid, psi_grid

def euler_generator(phi_grid: np.ndarray, psi_grid: np.ndarray):
    """Vectorized Euler's method for generating the current displacement and velocity profiles"""
    # get the number of grid points
    N = len(phi_grid) - 1

    for _ in range(num_steps-1):
        # only update the interior points -- the boundary points are fixed
        phi_grid[1:N] = phi_grid[1:N] + h * psi_grid[1:N]
        psi_grid[1:N] = psi_grid[1:N] + h * (nu/a)**2 * (phi_grid[2:N+1] + phi_grid[0:N-1] - 2*phi_grid[1:N]) 
        yield phi_grid, psi_grid

def animate(
    x_grid: np.ndarray, 
    phi_grid: np.ndarray, 
    psi_grid: np.ndarray, 
    display: bool = True, 
    save_gif: bool = False
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))
    
    # plot the displacement 
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.0005, 0.0005)
    ax1.set_title("Displacement")
    ax1.set_xlabel("Position (m)")
    ax1.set_ylabel("Displacement (m)")
    frame1, = ax1.plot([], [], lw=3)
    
    # plot for velocity 
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.2, 0.2)
    ax2.set_title("Velocity")
    ax2.set_xlabel("Position (m)")
    ax2.set_ylabel("Velocity (m/s)")
    frame2, = ax2.plot([], [], lw=3)
    
    frame_list = []
    frame_skip = 5  # skip every 5 frames to speed up the animation
    for i, (phi, psi) in enumerate(euler_generator(phi_grid, psi_grid)):
        if i % frame_skip == 0:
            frame1, = ax1.plot(x_grid, phi, "b")
            frame2, = ax2.plot(x_grid, psi, "r")
            frame_list.append([frame1, frame2])
    
    interval = 20  # increase the interval between frames (in milliseconds)
    anim = animation.ArtistAnimation(fig, frame_list, interval=interval, blit=True)
    
    plt.tight_layout()
    
    if display:
        plt.show()
    
    if save_gif:
        print("Saving animation...")
        fps = 1000 // interval  
        anim.save("wave.gif", writer="pillow", fps=fps)
        print("Animation saved as wave.gif")

if __name__ == "__main__":
    # get a grid of x values along the string
    # x_grid = np.linspace(0, L, N+1)
    x_grid = np.arange(0, L+a, a)    # so we can use a :) 

    # grid of initial displacement values -- all zeros
    # we initialize it as (num_steps, N+1) to store the values at each time step
    # phi_grid = np.zeros((num_steps, N+1), dtype=float)
    phi_grid = np.zeros(N+1, dtype=float) # for generator, we only need the current time step

    # set the initial velocity profile at t = 0 based on the x positions
    # psi_grid = np.zeros((num_steps, N+1), dtype=float)
    # psi_grid[0] = psi(x_grid, d, sigma, C, L)
    psi_grid = psi(x_grid, d, sigma, C, L) # for generator, we only need the current time step

    # solve for the displacement and velocity
    # disp, vel = euler(phi_grid, psi_grid, a, h, nu, num_steps)
    animate(x_grid, phi_grid, psi_grid, save_gif=False, display=True)
    # animate(x_grid, phi_grid, psi_grid, save_gif=True, display=False)