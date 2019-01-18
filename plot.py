import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys, glob
import numpy as np
import pdb

def readLog(file):
    with open(file) as f:
        lines = f.readlines()
    time = [l.strip().split()[-1] for l in lines if 'step time' in l]
    time = [float(t) for t in time]
    loss = [l.strip().split()[-1] for l in lines if 'remaining distance' in l]
    loss = [float(l) for l in loss]
    return time, loss

series = ['example2_localMax', 'example2_cg256', 'example2_cg64']
color = ['r', 'c', 'g']

fig, ax = plt.subplots(2)
for s,c in zip(series, color):
    files = glob.glob(s+'_trial*/log.txt')
    for f in files:
        time, loss = readLog(f)
        ax[0].plot(loss, color=c)
        time = np.cumsum(time)
        ax[1].plot(time, loss, color=c)
        ax[1].set_xlim(0, 12000)
plt.savefig('example2_limitCG.png')
