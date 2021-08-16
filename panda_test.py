import pybullet as p
import pybullet_data
import os
import time
p.connect(p.GUI)

time.sleep(10)

width=960
height=720
target_position=(0.7, 0.0, 0.70)
distance=.6
yaw=90
pitch=-35
roll=0
upAxisIndex=2



near = 0.1
far = 100.

view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=target_position,
                distance=distance,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                upAxisIndex=upAxisIndex,)
print(view_matrix)



proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=float(width) / height, nearVal=near, farVal=far)

print(proj_matrix)