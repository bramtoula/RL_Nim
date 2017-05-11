class SA():
    def __eliminateDuplicate(self, input):
        """
        Eliminate duplicate in a list.
        """
        output = []
        for x in input:
            if x not in output:
                output.append(x)
        return output
    
    def __init__(self, board):
        d = []
        d1 = [0] * len(board)
        while True:
            d.append(list(d1))
            
            if d[-1] == board:
                break
            
            d1[-1] += 1
            for m in range(len(board)):
                if d1[-1-m] > board[-1-m]:
                    d1[-1-m] = 0
                    d1[-2-m] += 1
        
        for i in range(len(d)):
            d[i].sort()
        d.sort()
        d = self.__eliminateDuplicate(d)
        
        states = {i:d[i] for i in range(len(d))}
        stringstates = [''.join(str(states[j])) for j in states]

        stateindex = {stringstates[i]:i for i in range(len(stringstates))}

        self.states = states
        self.stateindex = stateindex
        
        actions = []
        for i in range(len(board)):
            for j in range(board[i]):
                actions.append([i,j+1])

        self.actions = actions