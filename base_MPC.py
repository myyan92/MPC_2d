import os, sys, time
import numpy as np
import argparse
#import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import dynamics_inference.dynamic_models as dynamic_models
import pdb


class BaseMPC(object):
    def __init__(self, population, horizon, execute_step,
                 mental_dynamics,
                 explore_mode, node_selection, max_action,
                 explore_std_angle=0.0, explore_std_scale=0.0,
                 explore_std_zero=0.0, explore_std_one=0.0):
        self.population = population
        self.horizon = horizon
        self.execute_step = execute_step
        self.explore_mode = explore_mode
        self.node_selection = node_selection
        self.max_action = max_action
        self.std_angle = explore_std_angle
        self.std_scale = explore_std_scale
        self.std_zero = explore_std_zero
        self.std_one = explore_std_one
        if mental_dynamics == "physbam_2d":
            self.mental_dynamics = dynamic_models.physbam_2d()
        elif mental_dynamics == "physbam_3d":
            self.mental_dynamics = dynamic_models.physbam_3d()
        elif mental_dynamics == "neural":
            self.mental_dynamics = dynamic_models.neural_sim()
        else:
            raise ValueError("unrecognized mental dynamics type")

    @staticmethod
    def EuclideanToTangent(curve):
        diff_curve = np.zeros_like(curve)
        diff_curve[0:-1,:] = curve[1:,:]
        diff_curve[1:,:] -= curve[0:-1,:]
        tangent = np.arctan2(diff_curve[:,1], diff_curve[:,0])
        for i in range(1,len(tangent)):
            if tangent[i]-tangent[i-1] > np.pi:
                tangent[i] -= np.pi*2
            elif tangent[i]-tangent[i-1] < -np.pi:
                tangent[i] += np.pi*2
        return tangent

    def terminal_loss(self, current, goal, loss='Euclidean'):
        """Terminal loss function."""
        if loss == 'Euclidean':
            loss = np.sum(np.square(current-goal)) / current.shape[0]
        else:
            current_tangent = self.EuclideanToTangent(current)
            goal_tangent = self.EuclideanToTangent(goal)
            loss = np.sum(np.square(current_tangent-goal_tangent))
            loss /= current_tangent.shape[0]
        return loss

    def transient_loss(self, action, prev_action):
        """Transient loss function (action cost)."""
        loss = 0
        if action[0] != prev_action[0]:
            loss += 1e-6   # need regrasp
        else:
            a1 = np.array(action[1])
            a2 = np.array(prev_action[1])
            loss += 0.05*np.linalg.norm(a1-a2)**2
        return loss

    def evaluate(self, term_state, goal_state, actions, prev_action):
        """Calls terminal_loss and transient_loss to evaluate the trajectory."""
        loss = self.terminal_loss(term_state, goal_state)
#        for i in range(1,len(actions)):
#            loss += self.transient_loss(actions[i-1], actions[i])
#        if prev_action is not None:
#            loss += self.transient_loss(prev_action, actions[0])
        return loss


    def heuristic(self, current, goal, prev_plan):
        """Propose a promising trajectory based on current and goal state,
           optionally using previous plan.
        """
        # TODO: define the format of plan and return. Unify.
        # current and goal should be N*2 numpy array
        raise NotImplementedError()

    def add_exploration(self, init_actions, num_traj=1):
        """Add exploration noise to trajectory computed by heuristics.

        Args:
            init_actions: numpy array with shape [N,H,2].
            num_traj: Int. Number of trajectories to return for each traj in init.
            mode: One of "none", "independent" and "spline".
        Returns:
            actions: numpy array with shape [N*num_traj, H, 2].
        """
        init_actions = np.tile(init_actions, [num_traj, 1,1])
        if self.explore_mode == "none":
            return init_actions
        if self.explore_mode == "independent":
            noise = np.random.standard_normal(init_actions.shape)*np.array([[self.std_scale, self.std_angle]])
            actions_polar = np.zeros_like(init_actions)
            actions_polar[:,:,0]=np.linalg.norm(init_actions, axis=2)
            actions_polar[:,:,1]=np.arctan2(init_actions[:,:,0], init_actions[:,:,1])
            actions_polar += noise
            actions = np.zeros_like(actions_polar)
            actions[:,:,0]=actions_polar[:,:,0]*np.sin(actions_polar[:,:,1])
            actions[:,:,1]=actions_polar[:,:,0]*np.cos(actions_polar[:,:,1])
        else:
            # zero order noise is isotropic gaussian.
            # first order noise is perpendicular to trajectory direction
            zero_order_noise = np.random.standard_normal((init_actions.shape[0], 1,2))*self.std_zero
            first_order_noise = np.random.standard_normal((init_actions.shape[0], 1,1))*self.std_one
            traj_direction = np.sum(init_actions, axis=1, keepdims=True)
            first_order_direction = traj_direction[:,:,::-1]*np.array([-1.0, 1.0])
            first_order_direction /= np.linalg.norm(first_order_direction, axis=2, keepdims=True)
            actions = np.copy(init_actions)
            actions += zero_order_noise
            first_order_sine_wave = np.cos(np.linspace(0, np.pi, actions.shape[1]))
            actions += first_order_noise * first_order_direction * first_order_sine_wave.reshape((1,-1,1))
        return actions

    def aggregate(self, losses, actions):
        """Aggregate all evaluated trajectory to obtain the final plan."""
        raise NotImplementedError()

    def reset(self, start, goal):
        self.start = start
        self.goal = goal
        self.prev_act = None
        self.prev_init = None

    def plan(self):
        raise NotImplementedError()


