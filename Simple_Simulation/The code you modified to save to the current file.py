import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# 중력 상수 및 중심질량 등 상수 정의
G = 6.67e-11
mass_central = 1.0e30  # 중심 별의 질량, 태양 질량 기준 1.0e30 kg
n = 1e18

# 입자 수 및 초기 상태 설정
num_particles = 100000  # 입자 수
radii = np.random.uniform(0.5, 5.0, num_particles)  # 반지름 (거리)
angles = np.random.uniform(0, 2 * np.pi, num_particles)  # 각도
heights = np.random.uniform(-0.01, 0.01, num_particles)  # 높이
velocities = np.sqrt(G * mass_central / radii)  # 속도

positions = np.zeros((num_particles, 3))  # 입자들의 위치
vels = np.zeros((num_particles, 3))  # 입자들의 속도

# 입자 위치 및 속도 초기화
for i in range(num_particles):
    positions[i, 0] = radii[i] * np.cos(angles[i])
    positions[i, 1] = radii[i] * np.sin(angles[i])
    positions[i, 2] = heights[i]

    vels[i, 0] = -velocities[i] * np.sin(angles[i])
    vels[i, 1] = velocities[i] * np.cos(angles[i])
    vels[i, 2] = 0

# 시뮬레이션 파라미터
time_steps = 1000  # 시뮬레이션 시간 스텝 수
dt = 0.01  # 시간 간격
friction = 0.001  # 마찰 계수

# 입자의 속도 및 위치 업데이트 함수
def update_particles(positions, vels, dt, friction):
    for i in range(num_particles):
        r = np.linalg.norm(positions[i])
        accel = -G * mass_central / r**3 * positions[i]
        vels[i] += accel * dt
        vels[i] *= (1 - friction)
        positions[i] += vels[i] * dt
    return positions, vels

# 애니메이션을 위한 그림과 축 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 애니메이션 프레임 업데이트 함수
def animate(t):
    global positions, vels
    positions, vels = update_particles(positions, vels, dt, friction)
    ax.clear()
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=1)
    ax.scatter(0, 0, 0, color='r', s=100)  # 중심별
    ax.auto_scale_xyz(positions[:, 0], positions[:, 1], positions[:, 2])

# 애니메이션 생성
ani = animation.FuncAnimation(fig, animate, frames=time_steps, interval=50)

# 현재 디렉토리에 .gif 파일로 저장
current_dir = os.getcwd()
gif_path = os.path.join(current_dir, 'accretion_disk.gif')
ani.save(gif_path, writer='Pillow', fps=20)

# 결과 파일 경로 출력
print(f"Animation saved as: {gif_path}")
