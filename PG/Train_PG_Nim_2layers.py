""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import cPickle as pickle
import gym
import random
import os
import platform
from time import sleep


# hyperparameters
max_heap_nb = 5 # maximum number of heaps
max_heap_size = 4   # maximum items in one heap
H1 = 200 # number of hidden layer neurons # CHANGE
H2 = 200 # number of hidden layer neurons # CHANGE
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.9 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False
binary_input = False # True if we want to give the heaps as inputs represented in binary_input
opp_epsilon = 0.4 # The opponent will play opp_epsilon optimal
epsilon = 0.1 # Percentage of move the agent will take randomly

heap = []
originalHeap = []
heapNb = 0
heapMax = 0

# Functions from original Nim

# Define the command to clear the terminal (depends on the os...)
if platform.system() == 'Windows':
    def clear_terminal():
        os.system('cls')
else:
    def clear_terminal():
        os.system('clear')

def defineBoard():
    global heapNb
    global originalHeap
    global heapMax
    global heap
    global randomHeap
    os.system('clear')
    print "Let's start by defining our game:"

    randomHeap = 'y' == raw_input("Do you want the heaps to be random during learning ? (y/n)")
    if not randomHeap:
        heapNb = raw_input("Enter number of heaps you want: ")
        heapNb = int(heapNb)
        for x in range(1,int(heapNb)+1):
            num = raw_input("Enter number of matches on heap %d: " % x)
            heap.append(int(num))
        heap = fillHeapZeros(heap)
        originalHeap = list(heap)

    else:
        heap = defineRandomBoard()
    heap = sortHeap(heap)
    heapMax = max(heap)

# Function which fills the heap array with zeros if the number of heaps used is less than the maximum allowed
def fillHeapZeros(heap):
    while len(heap) < max_heap_nb:
        heap = np.append(heap,0)
    return heap

# Sorts heap in descending order
def sortHeap(heap):
    heap = np.sort(heap)
    heap[:] = heap[::-1]
    return heap

# Fills a random number of heaps with random number of items, within the maximum ranges
def defineRandomBoard():
    heap = []
    heapNb = random.randint(3,max_heap_nb)
    for i in range(0,heapNb):
        heap = np.append(heap,random.randint(1,max_heap_size+1))
    heap = fillHeapZeros(heap)
    heap = heap.astype(int)
    return heap

def printBoard(heap):
    clear_terminal()
    num = 0
    for num,row in enumerate(heap):
        print num+1,
        for match in range(0,row):
            print " |",
        print

def nimSum():
    return reduce(lambda x,y: x^y, heap)

def winningHeap():
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


def computerMove():
    if opp_epsilon > random.uniform(0, 1): # random move
        randomMove()
    else:
        if nimSum() == 0: # optimal move
            randomMove()
        else:
            heap[winningHeap()]^=nimSum()





# Returns the modified heap after a random play
def randomMove():
    global heap
    if np.amax(heap) == 0:
        return 0
    while True:
        play = random.randint(0,max_heap_nb*max_heap_size-1)
        actionRemoveIndex = int(play)/int(max_heap_size)
        actionRemoveNb = int(play)%int(max_heap_size)+1
        if heap[actionRemoveIndex] >= actionRemoveNb:
            heap[actionRemoveIndex] -= actionRemoveNb
            heap = sortHeap(heap)
            return play


def isItEnd():
    return all(z == 0 for z in heap)



defineBoard()
if binary_input:
    D = max_heap_nb*3 #CHANGE # input dimensionality: number of heapNb (in binary)
else:
    D = max_heap_nb
agentTurn = bool(random.getrandbits(1)) # Bool which represents player's turn. 1 is agent, 0 is computer opponent

# model initialization

if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H1,D) / np.sqrt(D) # "Xavier" initialization
    model['W2'] = np.random.randn(H1,H2) / np.sqrt(H2) # "Xavier" initialization
    model['W3'] = np.random.randn(H2,max_heap_nb*max_heap_size) / np.sqrt(H2)

grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def heap_to_binary(heap):
    x_bin = []
    for i in range (0,max_heap_nb):
        temp = ([int(d) for d in str(bin(heap[i]))[2:]])
        if len(temp) == 1:
            x_bin = np.append(x_bin,[0,0,temp[0]])
        elif len(temp) == 2:
            x_bin = np.append(x_bin,[0,temp[1],temp[0]])
        else:
            x_bin = np.append(x_bin,temp)
    return x_bin

def policy_forward(x):
    # Convert heaps in binary
    h1 = np.dot(model['W1'], x)
    h1[h1<0] = 0 # ReLU nonlinearity
    h2 = np.dot(model['W2'],h1)
    logp = np.dot(model['W3'].T, h2)
    p = sigmoid(logp)
    return p, h1, h2 # return probability of taking action 2, and hidden state

def policy_backward(eph1, eph2, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    # dW3 = np.dot(eph.T, epdlogp).ravel()
    dW3 = np.dot(eph2.T, epdlogp)
    dh2 = np.dot(epdlogp, model['W3'].T)
    dh2[eph2 <= 0] = 0 # backpro prelu
    dW2 = np.dot(dh2.T,eph1)
    dh1 = np.dot(dh2,model['W2'].T) # not sure
    dh1[eph1 <= 0] = 0 # backpro prelu
    dW1 = np.dot(dh1.T, epx)
    return {'W1':dW1, 'W2':dW2, 'W3':dW3}

xs,h1s,h2s,dlogps,drs = [],[],[],[],[]
running_reward = None
computerWin = False
playerWin = False
reward_sum = 0
episode_number = 0
while True:

    if render: printBoard(heap)

    # preprocess the observation, set input to network to be difference image
    # forward the policy network and sample an action from the returned probability
        # step the environment and get new measurements

    if not agentTurn: # computer turn
        computerMove()
        #   if max(heap) == 1:
        #     heap[heap.index(max(heap))]-= 1
        #   if max(heap) > 1:
        #     heap[heap.index(max(heap))]-=random.randint(1,max(heap)) # Change to total random play
        heap = sortHeap(heap)
        agentTurn = True
        continue

    if binary_input:
        x = heap_to_binary(heap)
    else:
        x = list(heap)
    reward = 0.0
    computerWin = isItEnd()
    aprob, h1, h2 = policy_forward(x)
    xs.append(x) # observation
    h1s.append(h1) # hidden state
    h2s.append(h2) # hidden state

    if epsilon > random.uniform(0, 1): # random move
        play = randomMove()
    else:
        play = 0
        for i in range(1,len(aprob)):  # Search biggest value in aprob for possible action
            actionRemoveIndex = int(i)/int(max_heap_size)
            actionRemoveNb = int(i)%int(max_heap_size)+1
            if (heap[actionRemoveIndex] >= actionRemoveNb) and (aprob[i] > aprob[play]):
                play = i

        if not computerWin:
            actionRemoveIndex = int(play)/int(max_heap_size)
            actionRemoveNb = int(play)%int(max_heap_size)+1
            heap[actionRemoveIndex] -= actionRemoveNb
            heap = sortHeap(heap)
            playerWin = isItEnd()
            agentTurn = False
    y = np.zeros(len(aprob))
    y[play] = 1.0

    dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

    done = playerWin or computerWin
    if computerWin:
        reward = -1.0
    elif playerWin:
        reward = +1.0

    drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)


    reward_sum += reward
    if done: # an episode finished
        episode_number += 1
        computerWin = False
        playerWin = False
        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        #epx = np.vstack(xs)
        #eph = np.vstack(hs)
        epx = np.vstack(xs)
        eph1 = np.vstack(h1s)
        eph2 = np.vstack(h2s)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs,h1s,h2s,dlogps,drs = [],[],[],[],[] # reset array memory
        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)

        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)
        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph1,eph2,epdlogp)
        for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k,v in model.iteritems():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        if episode_number % 1000 == 0:
            pickle.dump(model, open('save.p', 'wb'))
            print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)

        reward_sum = 0

        if randomHeap:
            heap = defineRandomBoard()
        else:
            heap = list(originalHeap)
        heap = sortHeap(heap)
        agentTurn = bool(random.getrandbits(1)) # Bool which represents player's turn. 1 is agent, 0 is computer opponent

    if reward != 0 and (episode_number % 1000 == 0) : # Nim has either +1 or -1 reward exactly when game ends.
        print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')
