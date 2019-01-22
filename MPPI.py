import os, sys, pdb, time
import numpy as np
from physbam_python.rollout_physbam import read_curve
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from MPC_2d.base_MPC import BaseMPC
import argparse, gin

@gin.configurable
class MPPI(BaseMPC):

    def __init__(self, inv_temperature, **kwargs):
        super().__init__(**kwargs)
        self.inv_temperature = inv_temperature

    def heuristic(self, err, horizon, prev_init=None):
        if prev_init is not None:
            err = err - np.sum(prev_init, axis=0)
            horizon = horizon-prev_init.shape[0]
        scale = np.linalg.norm(err)
        angle = np.arctan2(err[0],err[1])
        action_scale = [0.1 for _ in range(horizon)]
        for i in range(len(action_scale)):
            action_scale[i] = min(action_scale[i],scale)
            scale -= action_scale[i]
        actions = np.zeros((horizon,2))
        actions[:,0]=np.array(action_scale)*np.sin(angle)
        actions[:,1]=np.array(action_scale)*np.cos(angle)
        if prev_init is not None:
            actions = np.concatenate([prev_init, actions], axis=0)
        return actions

    def aggregate(self, losses, actions):
        losses = np.array(losses)*self.inv_temperature
        w = np.exp(-losses+np.amin(losses))
        w = w / np.sum(w)
        actions = np.array(actions)
        actions = np.tensordot(w, actions, axes=1)
        return actions

    def plan(self, current):
        if self.act_history:
           prev_act = self.act_history[-1]
        else:
           prev_act = None

        node_dists = np.linalg.norm(current-self.goal, axis=1)
        if self.node_selection == 'threshold':
            cand_nodes, = np.where(node_dists > np.amax(node_dists)*0.9)
            cand_nodes = cand_nodes.tolist()
        elif self.node_selection == 'localMax':
            cand_nodes = []
            for i in range(len(node_dists)):
                if (i==0 or node_dists[i] > node_dists[i-1]) and \
                   (i==len(node_dists)-1 or node_dists[i] > node_dists[i+1]):
                   if node_dists[i] > 0.01:
                        cand_nodes.append(i)
        if prev_act is not None and prev_act[0] not in cand_nodes:
            cand_nodes.append(prev_act[0])
        print("number of candidate nodes: ", len(cand_nodes))
        print("Candidate nodes:", cand_nodes)
        print("distances:", [node_dists[c] for c in cand_nodes])
        cand_seq = []
        for node in cand_nodes:
            if prev_act is not None and node==prev_act[0]:
                init_actions = self.heuristic(self.goal[node]-current[node], self.horizon, self.prev_init)
            else:
                init_actions = self.heuristic(self.goal[node]-current[node], self.horizon)
            # generate action sequencies from heuristics and evaluate
            sampled_actions_arr = self.add_exploration(init_actions, self.population)
            converted_actions = []
            for sa in sampled_actions_arr:
                converted_action = [(node, action) for action in sa]
                converted_actions.append(converted_action)
            term_states = self.mental_dynamics.execute_batch(current, converted_actions)
            losses = [self.evaluate(term_state, self.goal, actions, prev_act)
                      for term_state, actions in zip(term_states, converted_actions)]
            action_seq = self.aggregate(losses, sampled_actions_arr)
            cand_seq.append((node, action_seq))

        converted_actions = []
        for node, sa in cand_seq:
            converted_action = [(node, action) for action in sa]
            converted_actions.append(converted_action)
        term_states = self.mental_dynamics.execute_batch(current, converted_actions)
        losses = [self.evaluate(term_state, self.goal, actions, prev_act)
                  for term_state, actions in zip(term_states, converted_actions)]
        print("Losses for each cand node:", losses)
        idx = np.argmin(losses)
        final_act_seq = cand_seq[idx]
        self.prev_init = final_act_seq[1][self.execute_step:]
        final_act_seq = [(final_act_seq[0], ac) for ac in final_act_seq[1][:self.execute_step]]
        self.act_history.extend(final_act_seq)
        print(final_act_seq)
        return final_act_seq

    def reset(self, start, goal):
        super().reset(start, goal)
        self.prev_init = None

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

    planner = MPPI()
    planner.reset(start, goal)
    planner.main_loop()

    planner.save_animation('vis_control_MPPI.mp4')
