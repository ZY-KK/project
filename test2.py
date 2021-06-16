import pybullet as p
import pybullet_data
import numpy as np
import itertools
from task.Grasp.PandaGraspEnv import PandaGraspEnv
import time
'''
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
useMaximalCoordinates = False
a = p.loadURDF("plane.urdf", useMaximalCoordinates=useMaximalCoordinates)
#p.loadURDF("sphere2.urdf",[0,0,1])
b = p.loadURDF("cube.urdf", [0, 0, 0], useMaximalCoordinates=useMaximalCoordinates)
p.setGravity(0, 3, -10)
p.stepSimulation()
pts = p.getContactPoints(a,b)
print(a, b)
print("num pts=", len(pts))
totalNormalForce = 0
totalFrictionForce = [0, 0, 0]
totalLateralFrictionForce = [0, 0, 0]
normal = np.array([0.0, 0.0, 0.0])
for pt in pts:
        #print("pt.normal=",pt[7])
        #print("pt.normalForce=",pt[9])
        print(pt)
        normal+=pt[7]


# normal/=np.linalg.norm(normal)
# print(normal)
print(pts)
'''
'''
a = []

b = [1, 1, 1]
c = [2, 2, 2]
d = [3, 3, 3]
a.append(b)
a.append(c)
a.append(d)
print(a)
for n1, n2 in itertools.combinations(a, 2):
    print(n1, n2)
'''
'''
import pybullet as p
import time
import math

import pybullet_data
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.loadURDF("plane.urdf")
cubeId = p.loadURDF("cube_small.urdf", 0, 0, 1)
p.setGravity(0, 0, -10)
p.setRealTimeSimulation(1)
cid = p.createConstraint(cubeId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 1])
print(cid)
print(p.getConstraintUniqueId(0))
a = -math.pi
while 1:
  a = a + 0.01
  if (a > math.pi):
    a = -math.pi
  time.sleep(.01)
  p.setGravity(0, 0, -10)
  pivot = [a, 0, 1]
  orn = p.getQuaternionFromEuler([a, 0, 0])
  p.changeConstraint(cid, pivot, jointChildFrameOrientation=orn, maxForce=50)

p.removeConstraint(cid)
'''

'''
action = [0.2, 0.2, -1, 0.8, 1, 0.5, 0.5, 0.5]
tmp = np.asarray(action[1:4])
print(action)
constraint = [0.0, 0.5]
np.clip(tmp, constraint[0], constraint[1], out = tmp)
print(tmp)
action[1:4] = tmp
print(action)
'''

import gym

env = gym.make('PandaGraspEnv-v0')
env.reset()
for _ in range(1000):
    env.step(env.action_space.sample()) # take a random action
    time.sleep(1)
env.close()