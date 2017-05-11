import random as rnd
import numpy as np

class AgentSARSA():
    def __init__(self, SA, stepSize, discount, epsilon):
        self.state = 0
        self.action = 0

        self.actions = SA.actions
        self.states = SA.states
        self.stateindex = SA.stateindex

        self.Q = np.zeros([len(SA.states),len(SA.actions)])
        self.stepSize = stepSize
        self.discount = discount
        self.epsilon = epsilon

        self.optimalMovesMade = 0
        self.optimalMovesPossible = 0

    def readBoard(self, board):
        """ read board and return the state s"""
        b = ''.join(str(board))
        s = self.stateindex[b]
        return s

    def chooseAction(self, s):
        """ chooses action according to action policy """
        r = rnd.random()

        if r > self.epsilon: # for epsilon-greedy policy
            a = rnd.randrange(len(self.actions))
        else:       # choose best possible action in this state
            q = list(self.Q[s,:])
            m = max(q)
            if q.count(m) > 1: # if more than 1 action w/ max value
                bestAction = []
                for i in range(len(q)):
                    if q[i] == m:
                        bestAction.append(i)
                a = rnd.choice(bestAction)

            else:
                a = np.argmax(self.Q[s,:])
        if self.isValid(s, a) == False:
            return self.chooseAction(s)
        else:
            return a

    def changeBoard(self, board, a):
		action = self.actions[a]
		
		heap = action[0]
		amount = action[1]
		
		board[heap] -= amount
		board.sort()
		
		return board

    def move(self, board):
        
        s = self.readBoard(board)
        a = self.chooseAction(s)

        self.state = s
        self.action = a

        board = self.changeBoard(board, a)
		
        return board

    def winUpdate(self, R=1):
        s = self.state
        a = self.action
        
        self.Q[s][a] += self.stepSize*(R - self.Q[s][a])

    def updateQ(self, board, R=0):
        s = self.state
        a = self.action
        
        sp = self.readBoard(board)
        ap = self.chooseAction(sp)
        self.Q[s][a] += self.stepSize*(R + self.discount*self.Q[sp][ap] - self.Q[s][a])
		### NEED TO RETURN ap FOR NEXT STEP ###

    def loseUpdate(self, R=-1):
        s = self.state
        a = self.action
        self.Q[s][a] += self.stepSize*(R - self.Q[s][a])

    def isValid(self, s, a):
        action = self.actions[a]
        heap = action[0]
        amount = action[1]
        if (self.states[s][heap] - amount) < 0:
            self.Q[s][a] = -10
            return False
        else:
            return True

############### Optimal Policy  ##############

    def greedyMove(self, board):
                
        s = self.readBoard(board)
        a = self.greedyAction(s)

        board = self.changeBoard(board, a) 

        return board
        



    def greedyAction(self, s):
        # choose best possible action in this state
        q = list(self.Q[s,:])
        m = max(q)
        if q.count(m) > 1: # if more than 1 action w/ max value
            bestAction = []
            for i in range(len(q)):
                if q[i] == m:
                    bestAction.append(i)
            a = rnd.choice(bestAction)

        else:
            a = np.argmax(self.Q[s,:])
        if self.isValid(s, a) == False:
            return self.greedyAction(s)
        else:
            return a

    
        
        


        
