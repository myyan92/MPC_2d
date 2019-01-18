import os, sys
import numpy as np
from physbam_python.rollout_physbam import rollout_single, read_curve
from multiprocessing import Pool
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
        prob = np.exp(dist*5) / np.sum(np.exp(dist*5))
        node = np.random.choice(len(prob), p=prob)+1  # node is 1-based
        act = (goal[node-1,:]-current[node-1,:])
        act *= min(1.0, 0.15/np.linalg.norm(act))
        # add noise to act
        scale = np.random.uniform(0.8,1.2)
        angle = np.random.randn()*0.2  # std about 12 degree
        T = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])*scale
        act = np.dot(T,act)
        return (node, act)

    def parallel_plan_eval(self, cur_state, goal_state, horizon, prev_act):
        state = cur_state
        action = prev_act
        act_seq = []
        for t in range(horizon):
            action = self.heuristic(state, goal_state, action)
            state = self.execute(state, [action])
            act_seq.append(action)
        loss = self.evaluate(state, goal_state, act_seq, prev_act)
        return loss, act_seq

    def plan(self, current):
        if self.act_history:
           prev_act = self.act_history[-1]
        else:
           prev_act = None

        results = self.pool.starmap(self.parallel_plan_eval,
            [(current, self.goal, self.horizon, prev_act) for _ in range(self.population)])
        losses = [r[0] for r in results]
        cand_seq = [r[1] for r in results]
        idx = np.argmin(losses)
        final_act_seq = cand_seq[idx][:self.execute_step]
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
