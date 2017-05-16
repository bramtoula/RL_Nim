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

from collections import namedtuple
from itertools import count
from copy import deepcopy

#---------------------Initialize Game-----------------#
#Game parameters and state globals
heap = []
heaps = 4
heapMax = 5
steps_done = 0
maxBits = len(bin(heapMax))

#How random the AI we train against is
epsilon_rand = 1.0;

#Network learning parameters, also look at DQN construction
num_episodes = 100000
BATCH_SIZE = 128
REPLAY_SIZE = 10000
#learning rate?

USE_CUDA = torch.cuda.is_available()

#Reinforcement Learning parameters
GAMMA = 0.999
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 10000

#-------------GAME IMPLEMENTATION--------------------#

def resetBoard():
    global heap
    global done
    heap = []
    for x in range(1,int(heaps)+1):
        heap.append(x)#randint(1,5))
    done = 0

def nimSum():
    return reduce(lambda x,y: x^y, heap)

def winingHeap():
    return [x^nimSum() < x for x in heap].index(True)

def computerMove():
    global heap
    #removes randomly from largest heap
    if nimSum()==0 or random.randrange(1000)/1000.0 < epsilon_rand:
        heap[heap.index(max(heap))]-=randint(1,max(heap))
    else: heap[winingHeap()]^=nimSum()


def isItEnd():
    return all(z == 0 for z in heap)


#---------------DQN and Replay Memory Methods-----------------#
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(int(heaps), 32, True)
        self.linear2 = nn.Linear(32, 32, True)
        #self.linear3 = nn.Linear(32, 32, True)
        self.linear4= nn.Linear(32, int(heaps*heapMax), True)
    
        #self.linear1 = nn.Linear(int(heaps)*maxBits, 30, True)
        #self.linear2 = nn.Linear(30, 18, True)
        #self.linear3 = nn.Linear(18, 21, True)
        #self.linear4 = nn.Linear(21, 15, True)
        #self.linear5 = nn.Linear(15, 12, True)
        #self.linear6 = nn.Linear(12, int(heaps*heapMax), True)
    

    def forward(self, x):
        #convert heaps to binart=y
        inputLength = len(x)
        #newX = []
        #x_py = x.data.numpy();
        #x_py = np.sort(x_py)
        #for h in range(len(x_py)):
        #    intval = int(x_py[h])
        #    binaryRep = [int(digit) for digit in bin(intval)[2:]]
        #    binaryRep = np.lib.pad(binaryRep, (maxBits-len(binaryRep),0), 'constant', constant_values=(0,0))
        #    newX = newX + list(binaryRep)
        #x = Variable(torch.FloatTensor(newX))
        #x = x.view(int(inputLength/heaps), int(heaps)*maxBits)
        
        x = x.view(int(inputLength/heaps), int(heaps))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        #x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        return x



class Variable(autograd.Variable):

    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)


def getMaxValidAction():
    QSA_for_actions = model(Variable(torch.FloatTensor(heap), volatile=True)).data.cpu()
    curMax = -sys.maxint
    curActionIndex = -1

    index = 0
    for qsa in QSA_for_actions[0]:
        binNum = index/heapMax
        numPick = (index%heapMax)+1
        if qsa > curMax and heap[binNum] >= numPick:
            curActionIndex = index
            curMax = qsa
        index += 1
    return curActionIndex

#Actions are defined as bin*heapMax+numPickedUp
def select_action():
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    #exploration vs exploitation
    if sample > eps_threshold:
        return getMaxValidAction()
    else:
        #choose random valid heap and num
        nonZeroHeaps = []
        for x in range(heaps):
            if (heap[x] > 0):
                nonZeroHeaps.append(x);
        randBin = nonZeroHeaps[random.randrange(len(nonZeroHeaps))]
        randNum = random.randrange(heap[randBin])
        return randBin*heapMax+randNum


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
    if USE_CUDA:
        non_final_mask = non_final_mask.cuda()

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    #print loss
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



##################
# Testing method #
##################
def test():
    num_eps = 10000
    win_count = 0
    lose_count = 0
    
    total_moves = 0
    nimsum_moves = 0
    for i in range(num_eps):
        resetBoard()
        for t in count():
            action = select_action()
            current_heap = heap[:]
            binNum = action/heapMax
            amount = (action%heapMax)+1
            heap[binNum] -= amount
            
            total_moves += 1
            if nimSum() == 0:
                nimsum_moves += 1
            
            done = isItEnd()
            if done:
                win_count += 1
                break
            else:
                computerMove();
                done = isItEnd();
                if done:
                    lose_count += 1
                    break
            next_heap = heap[:]

    return win_count/float(num_eps), nimsum_moves/float(total_moves)


#################
# Training loop #
#################

resetBoard()

model = DQN()
memory = ReplayMemory(REPLAY_SIZE)
optimizer = optim.RMSprop(model.parameters())

if USE_CUDA:
    model.cuda()

print "TRAINING..."
actions_pushed = 0
for i_episode in range(1, num_episodes+1):
    steps_done+=1
    resetBoard()
    for t in count():
        current_heap = heap[:]
        action = select_action()
        #Update heap
        binNum = action/heapMax
        amount = (action%heapMax)+1
        heap[binNum] -= amount
        
        done = isItEnd()
        reward = torch.Tensor([0])
        if nimSum() == 0:
            reward = torch.Tensor([5])
        else:
            reward = torch.Tensor([-5])
        if done:
            #lost
            reward = torch.Tensor([10])
        else:
            #now ai move
            computerMove();
            done = isItEnd();
            #won
            if done:
                reward = torch.Tensor([-10])

        next_heap = torch.FloatTensor(heap[:])
        if done:
            next_heap = None

        memory.push(torch.FloatTensor(current_heap), torch.LongTensor([[action]]), next_heap, torch.FloatTensor(reward))
        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            if (i_episode % 1000 == 0):
                winP, opMoveP = test()
                print ["Epsiode", i_episode, winP, opMoveP]
            break



