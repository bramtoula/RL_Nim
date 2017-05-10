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
        for i in range(0,board[0]+1):
            for j in range(0,board[1]+1):
                for k in range(0,board[2]+1):
                    for l in range(0, board[3] + 1):
                        d.append(sorted([i,j,k,l]))

        d = self.__eliminateDuplicate(d)
        states = {i:d[i] for i in range(len(d))}
        stringstates = [''.join(str(states[j])) for j in states]

        stateindex = {stringstates[i]:i for i in range(len(stringstates))}

        self.states = states
        self.stateindex = stateindex
        
        actions = []
        for i in range(len(states)):
            temp = []
            for j in range(states[i][0]):
                temp.append([0,j+1])
            for j in range(states[i][1]):
                temp.append([1,j+1])
            for j in range(states[i][2]):
                temp.append([2,j+1])
            for j in range(states[i][3]):
                temp.append([3,j+1])
            actions.append(temp)

        self.actions = actions
