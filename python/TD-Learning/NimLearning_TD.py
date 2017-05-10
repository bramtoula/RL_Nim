########## Initialization ##########

#import numpy as np
import random as rnd
#import matplotlib.pyplot as plt

from SA import SA

from AgentQ import AgentQ
from AgentSARSA import AgentSARSA

# Variables initialization
# RL
stepSize = 0.1
discount = 1 # no discounting
epsilon = 0.1
# Nim
board_ini = [5,5,5,5]
sarsa_flag = False
runMax = 1E4
repetMax = 1

########## Reinforcement Learning ##########

board = board_ini
if sarsa_flag:
    agent = AgentSARSA(SA(board), stepSize, discount, epsilon)
else:
    agent = AgentQ(SA(board), stepSize, discount, epsilon)
episode = []
wins = []
optmoves = []
for repet in range(0,repetMax):
    for run in range(0,runMax):
        if run % 100 == 0:
            print("repet : {0}/{1}\n".format(repet, repetMax))
            print("run   : {0}/{1}\n".format(run, runMax))
    
        isFirst = rnd.randint(0,1)
        if isFirst == True: 
            while True: # a1 goes first
                if play1(board,end,a1) == False:
                    break
            board = board_ini
            
        if isFirst == False:
            c = 0
            while True: # comp goes first
                if play2(board,end,a1,c) == False:
                    break
                c += 1
            board = board_ini
            
        if j % interval  == 0: # Increase Epsilon over time
                    epslimit = 10000
                    a1.epsilon += interval*(1-epsilon)/epslimit
                    
                    x = 250 # Performance : play 100 games each 1000 episodes
                    a1.ngames = 0
                    a1.won = 0
                    a1.optimalMovesPossible = 0
                    a1.optimalMovesMade = 0
                    started = 0
                    for i in range(0,x):
                        r = rnd.randrange(2)
                        if r == 0:
                            started += 1
                            while True: # Agent first
                                if policyPlay1(board, end, a1) == False:
                                    break
                            board = [1,3,5,7]
                        if r == 1:
                            while True: # Computer first
                                if policyPlay2(board, end, a1) == False:
                                    break
                            board = [1,3,5,7]
                    
                    episode.append(j)
                    wins.append(a1.won/(x-started))
                    optmoves.append(a1.optimalMovesMade/a1.optimalMovesPossible)
            
def play1(board,end,Agent):
    """Agent vs Smart"""
    Agent.move(board)
    if board == end:
##        print(s,a)
        Agent.winUpdate(Agent.state,Agent.action,1)
        return False
    
    smartMove(board)
    if board == end:
        Agent.loseUpdate(-1)
        return False
    s = Agent.state
    a = Agent.action
    sp = Agent.readBoard(board)
    Agent.update(s,a,sp,0)

def play2(board, end , Agent,c):
    """ Smart vs Agent
Computer first """
    
    smartMove(board)
    if board == end:
        Agent.loseUpdate(-1)
        return False
    if c != 0:
        s = Agent.state
        a = Agent.action
        sp = Agent.readBoard(board)
        Agent.update(s,a,sp,0)
        
    Agent.move(board)
    if board == end:
        Agent.winUpdate(Agent.state,Agent.action,1)
        return False