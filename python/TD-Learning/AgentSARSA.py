import random as rnd
import numpy as np

class AgentSARSA():
    def __init__(self, SA, stepSize, discount, epsilon):
        self.ngames = 0
        self.won = 0

        self.state = 0
        self.action = 0

        self.actions = SA.actions
        self.states = SA.states
        self.stateindex = SA.stateindex

        self.Q = np.zeros([len(SA.states),len(SA.actions[len(SA.states)-1])])
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
            a = rnd.randrange(len(self.actions[s]))
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
		s = self.readBoard(board)
		action = self.actions[s][a]
		
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

    def winUpdate(self, s, a, R):
        self.won += 1
        self.Q[s][a] += self.stepSize*(R - self.Q[s][a])

    def update(self, s, a, sp, R):
        ap = self.chooseAction(sp)
        self.Q[s][a] += self.stepSize*(R + self.discount*self.Q[sp][ap] - self.Q[s][a])
		### NEED TO RETURN ap FOR NEXT STEP ###

    def loseUpdate(self, R):
        s = self.state
        a = self.action
        self.Q[s][a] += self.stepSize*(R - self.Q[s][a])

    def isValid(self, s, a):
        action = self.actions[s][a]
        heap = action[0]
        amount = action[1]
        if (self.states[s][heap] - amount) < 0:
            self.Q[s][a] = -10
            return False
        else:
            return True

############### Optimal Policy  ##############

    def policyMove(self, board):
                
        s = self.readBoard(board)
        a = self.policyAction(s)

        board = self.changeBoard(board, a) 

        return board
        



    def policyAction(self, s, to_avoid=[]):
        # choose best possible action in this state
        q = list(self.Q[s,:])
        for i in to_avoid:
            del q[i]
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
            return self.policyAction(s, to_avoid.append(a))
        else:
            return a

    
        
        


        
