import pybullet as p
import pybullet_data
import numpy as np
import itertools
'''
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
useMaximalCoordinates = False
a = p.loadURDF("plane.urdf", useMaximalCoordinates=useMaximalCoordinates)
#p.loadURDF("sphere2.urdf",[0,0,1])
b = p.loadURDF("cube.urdf", [0, 0, 0], useMaximalCoordinates=useMaximalCoordinates)
p.setGravity(0, 3, -10)
p.stepSimulation()
pts = p.getContactPoints(a)
print(a)
print("num pts=", len(pts))
totalNormalForce = 0
totalFrictionForce = [0, 0, 0]
totalLateralFrictionForce = [0, 0, 0]
normal = np.array([0.0, 0.0, 0.0])
for pt in pts:
        #print("pt.normal=",pt[7])
        #print("pt.normalForce=",pt[9])
    
        normal+=pt[7]


normal/=np.linalg.norm(normal)
print(normal)
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