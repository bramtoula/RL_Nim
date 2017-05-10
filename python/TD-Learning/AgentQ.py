import random as rnd
import numpy as np

class AgentQ():
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

        self.optimalMovesMade = 0 # number of optimal moves made
        self.optimalMovesPossible = 0 # number of optimal moves that were possible

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

    def optimalAction(self, sp, to_avoid=[]):
        q = list(self.Q[sp,:])
        for i in to_avoid:
            del q[i]
        m = max(q)
        if q.count(m) > 1: # if more than 1 action w/ max value
            bestAction = []
            for i in range(len(q)):
                if q[i] == m:
                    bestAction.append(i)

            ap = rnd.choice(bestAction)
        else:
            ap = np.argmax(self.Q[sp,:])
        
        if self.isValid(sp, ap) == False:
            return self.optimalAction(sp, to_avoid.append(ap))
        else:
            return ap
        

    def winUpdate(self, s, a, R):
        self.won += 1
        self.Q[s][a] += self.stepSize*(R - self.Q[s][a])

    def update(self, s, a, sp, R):
        ap = self.optimalAction(sp)
        self.Q[s][a] += self.stepSize*(R + self.discount*self.Q[sp][ap] - self.Q[s][a])

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


    def policyAction(self, sp, to_avoid=[]):
        # choose best possible action in this state
        q = list(self.Q[sp,:])
        for i in to_avoid:
            del q[i]
        m = max(q)
        if q.count(m) > 1: # if more than 1 action w/ max value
            bestAction = []
            for i in range(len(q)):
                if q[i] == m:
                    bestAction.append(i)

            ap = rnd.choice(bestAction)
        else:
            ap = np.argmax(self.Q[sp,:])
        
        if self.isValid(sp, ap) == False:
            return self.policyAction(sp, to_avoid.append(ap))
        else:
            return ap

    
        
        


        
