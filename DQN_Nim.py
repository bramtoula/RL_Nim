import os
from time import sleep
import random
import math
import sys
from random import randint


import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt

from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

#---------------------Initialize Game-----------------#
originalHeap = []
heap = []
heaps = int(0)
heapMax = 0

#Parameters to modify - also look at DQN construction
steps_done = 0
episode_durations = []
num_episodes = 10000
pauseTime = 0.0
epsilon_fail = 1.0;


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 1.0
EPS_END = 0.25
EPS_DECAY = 10
REPLAY_SIZE = 10000
USE_CUDA = torch.cuda.is_available()

#-------------GAME--------------------#
def defineBoard():
    global heaps
    global heapMax
    global originalHeap
    os.system('clear')
    print "Let's start by defining our game:"
    heaps = raw_input("Enter number of heaps you want: ")
    heaps = int(heaps)
    for x in range(1,int(heaps)+1):
        num = raw_input("Enter number of matches on heap %d: " % x)
        heap.append(int(num))
        heapMax = max(heapMax, int(num))
    originalHeap = list(heap)
    print

def resetBoard():
    global heap
    global done
    heap = originalHeap[:]
    done = 0

def printBoard(heap):
    os.system('clear')
    num = 0
    for num,row in enumerate(heap):
        print num+1,
        for match in range(0,row):
            print " |",
        print

def nimSum():
    return reduce(lambda x,y: x^y, heap)

def winingHeap():
    return [x^nimSum() < x for x in heap].index(True)

def userMove():
    row, num = raw_input("Enter row and num of matches you want to take separated with space ex.(1 2):  ").split()
    row, num = int(row)-1,int(num)
    
    try:
        if row <= -1: raise
        if num>0 and num<=heap[row]:
            heap[row]-=num
            printBoard(heap)
        else:
            printBoard(heap)
            print "WRONG NUMBER TRY AGAIN"
            userMove()
    except:
        printBoard(heap)
        print "WRONG ROW TRY AGAIN"
        userMove()
    if isItEnd(): print "YOU WIN"

def computerMove(disp):
    global heap
    sleep(pauseTime)
    #removes randomly from largest heap
    if nimSum()==0 or random.randrange(1000)/1000.0 < epsilon_fail:
        heap[heap.index(max(heap))]-=randint(1,max(heap))
    else: heap[winingHeap()]^=nimSum()
    if disp:
        printBoard(heap)


def isItEnd():
    return all(z == 0 for z in heap)


#while True:
#	userMove()
#	if isItEnd(): break
#	computerMove()
#	if isItEnd(): break

#---------------LEARNING-----------------#
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
        self.linear1 = nn.Linear(int(heaps), 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, int(heaps*heapMax))
    
    def forward(self, x):
        x = x.view(len(x)/int(heaps), int(heaps))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
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
        bin = index/heapMax
        numPick = (index%heapMax)+1
        if qsa > curMax and heap[bin] >= numPick:
            curActionIndex = index
            curMax = qsa
        index += 1
    return curActionIndex

#Actions are defined as bin*heapMax+numPickedUp
def select_action():
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
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

#################
# Training loop #
#################

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
    next_state_values.volatile = False
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

############
# MAIN CODE #
############

defineBoard()
printBoard(heap)

model = DQN()
memory = ReplayMemory(REPLAY_SIZE)
optimizer = optim.RMSprop(model.parameters())

if USE_CUDA:
    model.cuda()


print "TRAINING..."
actions_pushed = 0
for i_episode in range(num_episodes):
    #print i_episode
    global steps_done
    global heap
    steps_done = 0
    resetBoard()
    #printBoard(heap)
    for t in count():
        action = select_action()
        
        current_heap = heap[:]
        
        #print "RL Agent Turn"
        sleep(pauseTime)
        
        #Update heap
        bin = action/heapMax
        amount = (action%heapMax)+1
        heap[bin] -= amount
        #Display board- opyional
        #printBoard(heap)
        
        done = isItEnd()
        reward = torch.Tensor([0])
        if done:
            #lost
            reward = torch.Tensor([-1])
            #print "Agent Won"
        else:
            #print "AI Turn"
            #now ai move
            computerMove(0);
            #Get game state and reward
            done = isItEnd();
            #won
            if done:
                reward = torch.Tensor([1])
            #print "Agent Lost"
    
        next_heap = torch.FloatTensor(heap[:])
        if done:
            next_heap = None

        memory.push(torch.FloatTensor(current_heap), torch.LongTensor([[action]]), next_heap, torch.FloatTensor(reward))
        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            sleep(pauseTime)
            episode_durations.append(t + 1)
            break


print "TESTING..."
test = 1000
win_count = 0
lose_count = 0
for i in range(test):
    resetBoard()
    #printBoard(heap)
    for t in count():
        action = select_action()
        current_heap = heap[:]
        bin = action/heapMax
        amount = (action%heapMax)+1
        heap[bin] -= amount
        done = isItEnd()
        if done:
            win_count += 1
            break
        else:
            computerMove(0);
            done = isItEnd();
            if done:
                lose_count += 1
                break
        next_heap = heap[:]

print 'Win Percentage: '
print win_count/float(test)

