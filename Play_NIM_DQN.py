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

import matplotlib
import matplotlib.pyplot as plt

from collections import namedtuple
from itertools import count
from copy import deepcopy

#---------------------Initialize Game-----------------#
#Game parameters
heap = []
heaps = 4
heapMax = 5
maxBits = len(bin(heapMax))

#Parameters to modify - also look at DQN construction
pauseTime = 1.0
epsilon_rand = 1.0;

USE_CUDA = torch.cuda.is_available()
#-------------GAME IMPLEMENTATION--------------------#
def defineBoard():
    global heaps
    global heapMax
    global originalHeap
    os.system('clear')
    print "Let's start by defining our game:"
    for x in range(1,int(heaps)+1):
        num = raw_input("Enter number of matches on heap %d: " % x)
        heap.append(int(min(int(num), heapMax)))
    originalHeap = list(heap)
    print


def printBoard(heap):
    os.system('clear')
    num = 0
    print heap
    for num,row in enumerate(heap):
        print num+1,
        for match in range(heap[num]):
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
    if nimSum()==0 or random.randrange(1000)/1000.0 < epsilon_rand:
        heap[heap.index(max(heap))]-=randint(1,max(heap))
    else: heap[winingHeap()]^=nimSum()
    if disp:
        printBoard(heap)


def isItEnd():
    return all(z == 0 for z in heap)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(int(heaps)*maxBits, 32, True)
        self.linear2 = nn.Linear(32, 32, True)
        self.linear3 = nn.Linear(32, int(heaps*heapMax), True)

    def forward(self, x):
        #convert heaps to binart=y
        inputLength = len(x)
        newX = []
        x_py = x.data.numpy();
        for h in range(len(x_py)):
            intval = int(x_py[h])
            binaryRep = [int(digit) for digit in bin(intval)[2:]]
            binaryRep = np.lib.pad(binaryRep, (maxBits-len(binaryRep),0), 'constant', constant_values=(0,0))
            newX = newX + list(binaryRep)
        x = Variable(torch.FloatTensor(newX))
        x = x.view(int(inputLength/heaps), int(heaps)*maxBits)
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
        binNum = index/heapMax
        numPick = (index%heapMax)+1
        if qsa > curMax and heap[binNum] >= numPick:
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

defineBoard()
printBoard(heap)

model = DQN()
if USE_CUDA:
    model.cuda()


computer_player = 'DQN'
#computer_player = 'AI'

while not done
    userMove()
    action = select_action()
    
    done = isItEnd()
    if done:
        print "Agent Won"
    else:
        print "Agent Turn"
        if computer_player == 'AI':
            computerMove(0);
        else if computer_player == 'DQN':
            action = select_action()
            binNum = action/heapMax
            amount = (action%heapMax)+1
            heap[binNum] -= amount
            printBoard(heap)
        sleep(pauseTime)
        
        done = isItEnd();
        if done:
            print "You Won!"



