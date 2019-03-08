import os, sys
import numpy as np
from physbam_python.rollout_physbam_2d import read_curve
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from MPC_2d.base_MPC import BaseMPC
import argparse, gin

@gin.configurable
class VanillaMPC(BaseMPC):

    def heuristic(self, current, goal, prev_act):
        # current and goal should be N*2 numpy array
        dist = np.linalg.norm(current-goal, axis=1)
        if prev_act is not None:
            dist[prev_act[0]-1] *= 2  # encourange higher probability to sample smooth sequence
            # could also add interpolation between current action and previous action
        prob = np.exp(dist*50) / np.sum(np.exp(dist*50))
        node = np.random.choice(len(prob), p=prob)
        act = (goal[node,:]-current[node,:])
        act *= min(1.0, self.max_action/np.linalg.norm(act))
        # add noise to act
        scale = np.random.uniform(0.7,1.0)
        angle = np.random.randn()*0.2  # std about 12 degree
        T = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])*scale
        act = np.dot(T,act)
        return (node, act)

    def plan(self, current):
        cand_seqs = [[] for _ in range(self.population)]
        states = [current for _ in range(self.population)]
        actions = [self.prev_act for _ in range(self.population)]
        for t in range(self.horizon):
            next_actions = []
            for s,a in zip(states, actions):
                next_action = self.heuristic(s, self.goal, a)
                next_actions.append([next_action])
            for seq, na in zip(cand_seqs, next_actions):
                seq.extend(na)
            states = self.mental_dynamics.execute_batch(states, next_actions)
            actions = [na[0] for na in next_actions]
        losses = [self.evaluate(state, self.goal, actions, self.prev_act)
                  for state, actions in zip(states, cand_seqs)]

        idx = np.argmin(losses)
        final_act_seq = cand_seqs[idx][:self.execute_step]
        self.prev_act = final_act_seq[-1]
        print(final_act_seq)
        return final_act_seq

