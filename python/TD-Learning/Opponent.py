import random as rnd

class Opponent():
    def __init__(self, SA, policy="optimal", epsilon=0.1):        
        self.policy = policy
        self.epsilon = epsilon


    def readBoard(self, board):
        """ read board and return the state s"""
        b = ''.join(str(board))
        s = self.stateindex[b]
        return s
    
    def randomMove(self, board):
        randHeap = rnd.randrange(len(board)) # get heap
        
        if board[randHeap] == 0: # heap is empty
            return self.randomMove(board)
        else:
            randAct = rnd.randint(1,board[randHeap]) # get amount
            board[randHeap] -= randAct
            board.sort()
            return board
        
    def optimalMove(self, board):
        bestMoves = []
        
        for heap in range(len(board)):
            for action in range(1,1+board[heap]):
                temp_board = list(board)
                temp_board[heap] -= action
                          
                nimSum = 0
                for i in range(len(temp_board)):
                    nimSum ^= temp_board[i]
                
                if nimSum == 0:
                    bestMoves.append([heap,action])
        
        if bestMoves: # if an optimal move is available
            r = rnd.randrange(len(bestMoves))
            bestHeap = bestMoves[r][0]
            bestAction = bestMoves[r][1]
            
            board[bestHeap] -= bestAction
            board.sort()
            return board
        else:
            return self.randomMove(board)
        
    def e_optimalMove(self, board):
        r = rnd.random()
        
        if r < self.epsilon:
            return self.randomMove(board)
        else:
            return self.optimalMove(board)

    def move(self, board):
        if self.policy == "random":
            return self.randomMove(board)
        
        elif self.policy == "optimal":
            return self.optimalMove(board)
        
        elif self.policy == "e-optimal":
            return self.e_optimalMove(board)

    