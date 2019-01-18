from base_MPC import BaseMPC
import numpy as np
import matplotlib.pyplot as plt
import pdb

planner = BaseMPC(50, 10, 2, 0.001, 50, 0.2, 0.2)
planner.explore_mode="spline"
planner.std_zero = 1.5
planner.std_one = 1.0

init_actions = np.ones((1,10,2))*0.5
result = planner.add_exploration(init_actions, 10)
traj = np.copy(result)
for i in range(1,10):
    traj[:,i,:] += traj[:,i-1,:]

for i in range(10):
    plt.plot(traj[i,:,0], traj[i,:,1])
    plt.show()

pdb.set_trace()
result = planner.add_exploration(result, 3)
traj = np.copy(result)
for i in range(1,10):
    traj[:,i,:] += traj[:,i-1,:]

for i in range(10):
    plt.plot(traj[i,:,0], traj[i,:,1])
    plt.show()

