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

#######################################################
#-----------Initialize Game Variables-----------------#
#######################################################
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



######################################################
#-------------GAME IMPLEMENTATION--------------------#
######################################################
def resetBoard():
    global heap
    global done
    heap = []
    for x in range(1,int(heaps)+1):
        heap.append(randint(1,5))
        #heap.append(x)
        #heap.append(5)
    done = 0
    heap = list(np.sort(heap))

def nimSum(_heap):
    return reduce(lambda x,y: x^y, _heap)

def winingHeap(_heap):
    return [x^nimSum(_heap) < x for x in _heap].index(True)

#Computer with (sub)optimal strategy moves
def computerMove(ai_eps):
    global heap
    #removes randomly from largest heap
    if nimSum(heap)==0 or random.randrange(1000)/1000.0 < ai_eps:
        heap[heap.index(max(heap))]-=randint(1,max(heap))
    else: heap[winingHeap(heap)]^=nimSum(heap)
    heap = list(np.sort(heap))


def isItEnd():
    return all(z == 0 for z in heap)


###############################################################
#---------------DQN and Replay Memory Methods-----------------#
###############################################################
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

#Stores the memory of state, action and rewards in play history
#We will sample from to construct our batch when training the network
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

#Our neural network used to represent the Q function
#Here we use just 1 layer with 32 hidden neurons to perform our paramter grid search
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(int(heaps), 32, True)
        self.linear2= nn.Linear(32, int(heaps*heapMax), True)

    def forward(self, x):
        inputLength = len(x)
        x = x.view(int(inputLength/heaps), int(heaps))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

#Returns the best action index given curHeap
def getMaxValidAction(curHeap):
    curHeap = list(np.sort(curHeap))
    QSA_for_actions = model(Variable(torch.FloatTensor(curHeap), volatile=True)).data.cpu()
    curMax = -sys.maxint
    curActionIndex = -1

    index = 0
    for qsa in QSA_for_actions[0]:
        binNum = index/heapMax
        numPick = (index%heapMax)+1
        if qsa > curMax and curHeap[binNum] >= numPick:
            curActionIndex = index
            curMax = qsa
        index += 1
    return curActionIndex

#Returns the action index, sometimes choosing a move randomly
def select_action(greedy_eps):
    global heap
    sample = random.random()
    eps_threshold = greedy_eps
    #exploration vs exploitation
    if sample > eps_threshold:
        return getMaxValidAction(heap)
    else:
        #choose random valid heap and num
        nonZeroHeaps = []
        for x in range(heaps):
            if (heap[x] > 0):
                nonZeroHeaps.append(x);
        randBin = nonZeroHeaps[random.randrange(len(nonZeroHeaps))]
        randNum = random.randrange(heap[randBin])
        return randBin*heapMax+randNum

#The RL agent moves
def agentMove(greedy_eps):
    global heap
    action = select_action(greedy_eps)
    binNum = action/heapMax
    amount = (action%heapMax)+1
    heap[binNum] -= amount
    heap = list(np.sort(heap))
    return action

#Optimizes the weights of the neural network
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

    # Don't consider end game states for next state values since there is no state after the finish
    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
    if USE_CUDA:
        non_final_mask = non_final_mask.cuda()

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    next_state_values.volatile = False
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



#################################################
#---------------Testing Methods-----------------#
#################################################
#Play against an AI a bunch of times
def test(ai_eps, lr, greedy_eps):
    num_eps = 10000
    win_count = 0
    lose_count = 0

    total_moves = 0
    nimsum_moves = 0
    for i in range(num_eps):
        resetBoard()
        firstAITurn = randint(0,1)
        if (firstAITurn == 1):
            computerMove(ai_eps);
        for t in count():
            agentMove(greedy_eps)
            total_moves += 1
            if nimSum(heap) == 0:
                nimsum_moves += 1

            done = isItEnd()
            if done:
                win_count += 1
                break
            else:
                computerMove(ai_eps);
                done = isItEnd();
                if done:
                    lose_count += 1
                    break
            next_heap = heap[:]

    return win_count/float(num_eps), nimsum_moves/float(total_moves)


#Out of all the states where an optimal move is possible, return how many the
#agent correctly chooses an optimal move
def getOptimalMovePercentage(model):
    nimSumMovePossible = 0;
    nimSumMove = 0;
    for h0 in range(0, heapMax+1):
        for h1 in range(h0, heapMax+1):
            for h2 in range (h1, heapMax+1):
                for h3 in range (h2, heapMax+1):
                    curHeap = [h0, h1, h2, h3];
                    if (sum(curHeap) == 0 or nimSum(curHeap) == 0):
                        continue
                    nimSumMovePossible += 1;
                    #Get q function values for state
                    QSA_for_actions = model(Variable(torch.FloatTensor(curHeap), volatile=True)).data.cpu()

                    index = getMaxValidAction(curHeap)
                    binNum = index/heapMax
                    numPick = (index%heapMax)+1
                    curHeap[binNum] -= numPick
                    if (nimSum(curHeap) == 0):
                        nimSumMove += 1;

    return float(nimSumMove)/nimSumMovePossible;

#Call this to calculate the percentage of times a completely random player would choose the optimal move
#As in getOptimalMovePercentage, only considers game states where an optimal move is possible
def getOptimalMovePercentageForRandom():
    pRandomNimSum = 0
    totalValidStates = 0;
    for h0 in range(0, heapMax+1):
        for h1 in range(h0, heapMax+1):
            for h2 in range (h1, heapMax+1):
                for h3 in range (h2, heapMax+1):
                    curHeap = [h0, h1, h2, h3];
                    if (sum(curHeap) == 0 or nimSum(curHeap) == 0):
                        continue
                    totalValidStates += 1
                    
                    movesPossible = 0;
                    nimSumMovesPossible = 0;
                    heapTest = curHeap[:]
                    
                    for index in range(heapMax*heaps):
                        heapTest = curHeap[:]
                        binNum = index/heapMax
                        numPick = (index%heapMax)+1
                        if heapTest[binNum] >= numPick:
                            movesPossible += 1
                            heapTest[binNum] -= numPick
                            if (nimSum(heapTest) == 0):
                                nimSumMovesPossible += 1
                    pRandomNimSum += float(nimSumMovesPossible)/movesPossible;

    return pRandomNimSum/totalValidStates



#################################################
#---------------Main Grid Seach-----------------#
#################################################
#print getOptimalMovePercentageForRandom()
OptimalMoveArray = np.zeros((len(epsilon_rand), len(LEARNING_RATE), len(STATIC_EPS)))

#Perform grid search over parameter space and calculate percentage of optimal moves made for each resulting model
ai_eps_ind = -1
for ai_eps in epsilon_rand:
    ai_eps_ind += 1
    lr_ind = -1
    for lr in LEARNING_RATE:
        lr_ind += 1
        greedy_eps_ind = -1
        for greedy_eps in STATIC_EPS:
            greedy_eps_ind += 1
            print ["TRAINING...", "AI Sub-Optimality:", ai_eps, "Learning Rate:", lr, "Epsilon Greedy:", greedy_eps]
            resetBoard()
            model = DQN()
            memory = ReplayMemory(REPLAY_SIZE)
            optimizer = optim.RMSprop(model.parameters(), lr)

            if USE_CUDA:
                model.cuda()

            actions_pushed = 0
            #Run num_episode trials, optimizing the model after each trial
            for i_episode in range(1, num_episodes+1):
                steps_done+=1
                resetBoard()
                firstAITurn = randint(0,1)
                if (firstAITurn == 1):
                    computerMove(ai_eps);
                for t in count():
                    current_heap = heap[:]
                    action = agentMove(greedy_eps)
                    done = isItEnd()
                    reward = torch.Tensor([0])
                    if done:
                        reward = torch.Tensor([10])
                    else:
                        computerMove(ai_eps);
                        done = isItEnd();
                        if done:
                            reward = torch.Tensor([-10])

                    next_heap = torch.FloatTensor(heap[:])
                    if done:
                        next_heap = None
                    
                    #push state, action and reward into memory
                    memory.push(torch.FloatTensor(current_heap), torch.LongTensor([[action]]), next_heap, torch.FloatTensor(reward))
                    # Perform one step of the optimization
                    optimize_model()
                    if done:
                        break

            #winP, opMoveP = test(ai_eps, lr, 1.0)
            opMoveP = getOptimalMovePercentage(model)
            print ["Optimal Move Percent:", opMoveP]
            OptimalMoveArray[ai_eps_ind, lr_ind, greedy_eps_ind] = opMoveP;
            sys.stdout.flush()

np.save('./grid_optimal', OptimalMoveArray)


###############################################
#-----------Plot the results array------------#
###############################################
##Uncomment to plot results
#np.load('./grid_optimal.npy', OptimalMoveArray)
#for i in range(len(epsilon_rand)):
#    plt.imshow(OptimalMoveArray[i,:,:].T, origin='lower', extent=(LEARNING_RATE[0], LEARNING_RATE[-1], STATIC_EPS[0], STATIC_EPS[-1]), \
               vmin=0., vmax=1., interpolation='none', cmap='hot')
#    cbar = plt.colorbar(); cbar.set_label("Optimality Measure");
#    plt.xlabel("Step size (alpha)"); plt.ylabel("Epsilon (for the learning policy)")
#    plt.title("Opponent optimal at {:.1f}%".format((1.-epsilon_rand[i])*100.))
#    plt.show()
