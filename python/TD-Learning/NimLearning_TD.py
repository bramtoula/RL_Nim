########## Initialization ##########

#↨import numpy as np
import random as rnd
import matplotlib.pyplot as plt

from SA import SA

from AgentQ import AgentQ
from AgentSARSA import AgentSARSA
from Opponent import Opponent

# Variables initialization
# RL
stepSize = 0.1
discount = 1 # no discounting
epsilon = 0.1 # for the e-greedy policy
# Nim
board_ini = sorted([5,5,5,5])
sarsa_flag = False
runMax = int(4E4)
repetMax = int(1)

# Function initialization
def init_board():
    """
    Return a random board based on board_ini
    """
    board = list(board_ini)
    for i in range(len(board_ini)):
        board[i] = rnd.randint(0,board_ini[i])
    board.sort()
    
    if board[-1] == 0:
        return init_board()
    return board


########## Reinforcement Learning ##########
# Board and agent
board = list(board_ini)
board_end = [0] * len(board_ini)
if sarsa_flag:
    agent = AgentSARSA(SA(board), stepSize, discount, epsilon)
else:
    agent = AgentQ(SA(board), stepSize, discount, epsilon)
oppLearning = Opponent(SA(board), policy="e-optimal", epsilon=0.1)
oppOptimal = Opponent(SA(board), policy="optimal")

# Learning curves parameters
learning_win = []
greedy_win = []
optimalMoves = []
optimalMoves_runNb = []

# Learning
for repet in range(repetMax):
    for run in range(runMax):
        if (run+1) % 100 == 0:
            print("repet : {0}/{1}".format(repet+1, repetMax))
            print("run   : {0}/{1}\n".format(run+1, runMax))
        
        board = init_board()
        
        agentIsFirst = rnd.randint(0,1)
        if agentIsFirst == False:
            oppLearning.move(board)
            if board == board_end:
                learning_win.append(0)
                continue
        
        while True:
            agent.move(board)
            if board == board_end:
                agent.winUpdate()
                learning_win.append(1)
                break
            
            oppLearning.move(board)
            if board == board_end:
                agent.loseUpdate()
                learning_win.append(0)
                break
                
            agent.updateQ(board)
        
        ### Test the agent every 100 runs on 100 more runs
        if (run+1) % 100 == 0:
            optMovePossible = 0.
            optMoveMade = 0.
            
            for _ in range(100):
                board = init_board()
                    
                agentIsFirst = rnd.randint(0,1)
                if agentIsFirst == False:
                    oppOptimal.move(board)
                    if board == board_end:
                        greedy_win.append(0)
                        continue
                
                while True:
                    before = 0
                    for i in range(len(board)):
                        before ^= board[i]
                    if before != 0:
                        optMovePossible += 1
                        
                    agent.greedyMove(board)
                   
                    after = 0
                    for i in range(len(board)):
                        after ^= board[i]
                    if after == 0:
                        optMoveMade += 1
                    
                    if board == board_end:
                        greedy_win.append(1)
                        break
                    
                    oppOptimal.move(board)
                    if board == board_end:
                        greedy_win.append(0)
                        break
            optimalMoves.append(optMoveMade/optMovePossible*100)
            optimalMoves_runNb.append(run)



########## Learning curves ##########

# Window averaging
half_window = 1000
learning_win_ave = []
greedy_win_ave = []

for i in range(len(learning_win)):
    startIndex = i - half_window
    if startIndex < 0:
        startIndex = 0
        
    endIndex = i + half_window + 1
    if endIndex > len(learning_win):
        endIndex = len(learning_win)
    
    learning_win_ave.append(float(sum(learning_win[startIndex:endIndex])) / (len(learning_win[startIndex:endIndex])))

for i in range(len(greedy_win)):
    startIndex = i - half_window
    if startIndex < 0:
        startIndex = 0
        
    endIndex = i + half_window + 1
    if endIndex > len(learning_win):
        endIndex = len(learning_win)
    
    greedy_win_ave.append(float(sum(greedy_win[startIndex:endIndex])) / (len(greedy_win[startIndex:endIndex])))

plt.plot(learning_win_ave)
plt.plot(greedy_win_ave)
plt.legend(["Learning", "Greedy"])
plt.title("Learning curves")
plt.show() 

plt.plot(optimalMoves_runNb, optimalMoves)
plt.title("Percentage of optimal moves done")
plt.show() 
    


########## Test of the agent after learning ##########
trials = 2000
wins = 0
winStart = 0
optMove = 0
optDone = 0
for i in range(trials):
    board = init_board()
        
    agentIsFirst = rnd.randint(0,1)
    if agentIsFirst == False:
        oppOptimal.move(board)
        if board == board_end:
            continue
    
    before = 0
    for i in range(len(board)):
        before ^= board[i]
    if before != 0:
        winStart += 1
    while True:
        before = 0
        for i in range(len(board)):
            before ^= board[i]
        if before != 0:
            optMove += 1
            
        agent.greedyMove(board)
       
        after = 0
        for i in range(len(board)):
            after ^= board[i]
        if after == 0:
            optDone += 1
        
        if board == board_end:
            wins += 1
            break
        
        oppOptimal.move(board)
        if board == board_end:
            break

print "---"
print "wins = {}/{} = {:.2f}%\noptDone = {}\noptMove = {}\nopt = {:.2f}%".format(wins, winStart, float(wins)/float(winStart)*100, \
              optDone, optMove, float(optDone)/float(optMove)*100)










