import os, sys
import numpy as np
from physbam_python.rollout_physbam import read_curve
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
        if self.act_history:
           prev_act = self.act_history[-1]
        else:
           prev_act = None

        cand_seqs = [[] for _ in range(self.population)]
        states = [current for _ in range(self.population)]
        actions = [prev_act for _ in range(self.population)]
        for t in range(self.horizon):
            next_actions = []
            for s,a in zip(states, actions):
                next_action = self.heuristic(s, self.goal, a)
                next_actions.append([next_action])
            for seq, na in zip(cand_seqs, next_actions):
                seq.extend(na)
            states = self.mental_dynamics.execute_batch(states, next_actions)
            actions = [na[0] for na in next_actions]
        losses = [self.evaluate(state, self.goal, actions, prev_act)
                  for state, actions in zip(states, cand_seqs)]

        idx = np.argmin(losses)
        final_act_seq = cand_seqs[idx][:self.execute_step]
        self.act_history.extend(final_act_seq)
        print(final_act_seq)
        return final_act_seq

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('start', help="starting state txt file")
    parser.add_argument('goal', help="goal state txt file")
    parser.add_argument('--gin_config', default='', help="path to gin config file.")
    parser.add_argument('--gin_bindings', action='append', help='gin bindings strings.')
    args = parser.parse_args()

    gin.parse_config_files_and_bindings([args.gin_config], args.gin_bindings)

    start = read_curve(args.start)
    goal = read_curve(args.goal)

    planner = VanillaMPC()
    planner.reset(start, goal)
    planner.main_loop()

    planner.save_animation('vis_control.mp4')
