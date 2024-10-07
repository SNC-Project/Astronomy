"""
Simple Dump of codes about Accretion Disk around the star. To use colab, the result will be .gif file.
It takes a long time. So change the parameter(espicially num_particles & time_steps). num_particles ~25000 / time_steps ~500 or ~100
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

G = 6.67e-11
mass_central = 1.0e30  # Mass of Star. For sun, M ~ 1.0e30 kg
n = 1e18

num_particles = 100000    # # of particles around the star.
radii = np.random.uniform(0.5, 5.0, num_particles)
angles = np.random.uniform(0, 2 * np.pi, num_particles)
heights = np.random.uniform(-0.01, 0.01, num_particles)
velocities = np.sqrt(G * mass_central / radii)

positions = np.zeros((num_particles, 3))
vels = np.zeros((num_particles, 3))

for i in range(num_particles):
    positions[i, 0] = radii[i]*np.cos(angles[i])
    positions[i, 1] = radii[i]*np.sin(angles[i])
    positions[i, 2] = heights[i]
    
    vels[i, 0] = -velocities[i] * np.sin(angles[i])
    vels[i, 1] = velocities[i] * np.cos(angles[i])
    vels[i, 2] = 0

time_steps = 10000 # 관측 횟수
dt = 0.01
friction = 0.001 # 마찰 계수(입자들 간의 마찰로 인한 변위를 확인하기 위함.)

def update_particles(position, vels, dt, friction):
    for i in range(num_particles):
        r = np.linalg.norm(position[i])
        accel = -G*mass_central/r**3 * position[i]
        
        vels[i] += accel*dt
        vels[i] *= (1 - friction)
        
        position[i] += vels[i] * dt
        
    return position, vels



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def animate(t):
    global positions, vels
    positions, vels = update_particles(positions, vels, dt, friction)
    ax.clear()
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=1)
    ax.scatter(0, 0, 0, color='r', s=100)
    ax.auto_scale_xyz(positions[:, 0], positions[:, 1], positions[:, 2])

ani = animation.FuncAnimation(fig, animate, frames=time_steps, interval=50)

ani.save('accretion_disk.gif', writer='Pillow', fps=20)

from IPython.display import Image
Image(filename='accretion_disk.gif')
