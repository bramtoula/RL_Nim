import os
from time import sleep
import random
import math
import sys
from random import randint


import argparse
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import matplotlib.pyplot as plt

from collections import namedtuple
from itertools import count
from copy import deepcopy

random.seed(52)
#Game parameters and state globals
heap = []
heaps = 4
heapMax = 5
steps_done = 0
maxBits = len(bin(heapMax))

#How random the AI we train against is
epsilon_rand = np.linspace(0,1,4)

#Network learning parameters
num_episodes = 10000
BATCH_SIZE = 128
REPLAY_SIZE = 10000

USE_CUDA = False

#Reinforcement Learning parameters
GAMMA = 1.0

LEARNING_RATE = np.linspace(0,1,11)
STATIC_EPS = np.linspace(0,1,11)

###############################################
#-----------Plot the results array------------#
###############################################
#Uncomment to plot results as well
#OptimalMoveArray = np.load('./grid_optimal.npy')
#for i in range(len(epsilon_rand)):
#    plt.imshow(OptimalMoveArray[i,:,:].T, origin='lower', extent=(LEARNING_RATE[0], LEARNING_RATE[-1], STATIC_EPS[0], STATIC_EPS[-1]), \
               vmin=0., vmax=1., interpolation='none', cmap='hot')
#    cbar = plt.colorbar(); cbar.set_label("Optimality Measure");
#    plt.xlabel("Step size (alpha)"); plt.ylabel("Epsilon (for the learning policy)")
#    plt.title("Opponent optimal at {:.1f}%".format((1.-epsilon_rand[i])*100.))
#    plt.show()
