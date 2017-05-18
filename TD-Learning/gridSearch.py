########## Initialization ##########
# import general librairies
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from IPython.display import clear_output
# personal librairies
from SA import SA
from AgentQ import AgentQ
from Opponent import Opponent

# Variables initialization
# RL
discount = 1 # no discounting (gamma)
stepSize = [i for i in np.linspace(0,1,11)] # alpha
epsilon =  [i for i in np.linspace(0,1,11)] # for the e-greedy policy
opp_epsilon = [i for i in np.linspace(0,1,4)] # for the e-optimal opponent
# Nim
board_ini = sorted([5,5,5,5]) # Biggest board for learning Nim
runMax = int(1E4) # Number of runs for the learning

# Function initialization
def init_board():
    """
    Return a random board based on board_ini
    """
    for i in range(len(board_ini)):
        board[i] = rnd.randint(0,board_ini[i])
    board.sort()
    
    if board[-1] == 0:
        return init_board()
    return board


########## Grid search ##########
# Board and agent
board = list(board_ini)
board_end = [0] * len(board_ini)
optMoveFound_gridSearch = np.zeros((len(stepSize),len(epsilon),len(opp_epsilon)))
searchNb = 0

# Learn for each set of parameters
for ii in range(len(stepSize)):
    for jj in range(len(epsilon)):
        for kk in range(len(opp_epsilon)):
            searchNb += 1
            
            agent = AgentQ(SA(board), stepSize[ii], discount, epsilon[jj])
            oppLearning = Opponent(SA(board), policy="e-optimal", epsilon=opp_epsilon[kk])
            
            # Learning
            for run in range(runMax):
                if (run+1) % 1000 == 0:
                    clear_output()
                    print("search: {}/{}\n".format(searchNb, len(stepSize)*len(epsilon)*len(opp_epsilon)))
                    print("run   : {}/{}\n\n".format(run+1, runMax))
                
                board = init_board()
                
                agentIsFirst = rnd.randint(0,1)
                if agentIsFirst == False:
                    oppLearning.move(board)
                    if board == board_end:
                        continue
                
                while True:
                    agent.move(board)
                    if board == board_end:
                        agent.winUpdate()
                        break
                    
                    oppLearning.move(board)
                    if board == board_end:
                        agent.loseUpdate()
                        break
                        
                    agent.updateQ(board)
             
            # Compute the F-score after the learning for that particular set of parameters
            optMove_P = 0.
            optMove_TP = 0.
            optMove_FP = 0.
            for s in agent.states:
                board = list(agent.states[s])
                for heap in range(len(board)):
                    for action in range(1,1+board[heap]):
                        temp_board = list(board)
                        temp_board[heap] -= action
                                  
                        nimSum = 0
                        for i in range(len(temp_board)):
                            nimSum ^= temp_board[i]
                        
                        a = agent.actions.index([heap,action])
                        if nimSum == 0:
                            optMove_P += 1.
                            if agent.Q[s][a] >= 0.9:
                                optMove_TP += 1.
                        elif agent.Q[s][a] >= 0.9:
                            optMove_FP += 1.
            
            optMoveFound_Recall = optMove_TP/optMove_P
            if optMove_TP+optMove_FP == 0.:
                optMoveFound_Precision = 0.
            else:
                optMoveFound_Precision = optMove_TP/(optMove_TP+optMove_FP)
            if optMoveFound_Precision+optMoveFound_Recall == 0:
                optMoveFound_F = 0.
            else:
                optMoveFound_F = 2*optMoveFound_Precision*optMoveFound_Recall / \
                                  (optMoveFound_Precision+optMoveFound_Recall)
            
            optMoveFound_gridSearch[ii,jj,kk] = optMoveFound_F


########## Plot of the results ##########
# Plot multiple 2D graphs
for i in range(len(opp_epsilon)):
    plt.imshow(optMoveFound_gridSearch[:,:,i].T, origin='lower', extent=(stepSize[0], stepSize[-1], epsilon[0], epsilon[-1]), \
               vmin=0., vmax=1., interpolation='none', cmap='hot')
    cbar = plt.colorbar(); cbar.set_label("F-Score")
    plt.xlabel("Step size (= alpha)"); plt.ylabel("Epsilon (exploration term)")
    plt.title("Opponent optimal at {:.1f}%".format((1.-opp_epsilon[i])*100.))
    plt.savefig("figures/gridSearch_{}.pdf".format(i))
    plt.show()

# Look for the first best set of parameters
index_best = np.unravel_index(np.argmax(optMoveFound_gridSearch), optMoveFound_gridSearch.shape)
print "The optimal parameters are found to be:"
print "step size = {}".format(stepSize[index_best[0]])
print "epsilon = {}".format(epsilon[index_best[1]])
print "opp_epsilon = {}".format(opp_epsilon[index_best[2]])

