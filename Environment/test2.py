import pybullet as p
import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
useMaximalCoordinates = False
a = p.loadURDF("plane.urdf", useMaximalCoordinates=useMaximalCoordinates)
#p.loadURDF("sphere2.urdf",[0,0,1])
b = p.loadURDF("cube.urdf", [0, 0, 0], useMaximalCoordinates=useMaximalCoordinates)
p.setGravity(0, 3, -10)
p.stepSimulation()
pts = p.getContactPoints(b)
print(a, b)
print("num pts=", len(pts))
totalNormalForce = 0
totalFrictionForce = [0, 0, 0]
totalLateralFrictionForce = [0, 0, 0]
for pt in pts:
        #print("pt.normal=",pt[7])
        #print("pt.normalForce=",pt[9])
    print(pt)