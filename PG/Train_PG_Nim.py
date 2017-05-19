""" Trains an agent with (stochastic) Policy Gradients on Nim. """
import numpy as np
import cPickle as pickle
import random
import os
import platform
import matplotlib.pyplot as plt
from time import sleep


# Fixed hyperparameters
max_heap_nb = 4 # maximum number of heaps
max_heap_size = 5   # maximum items in one heap
H1 = 32 # number of hidden layer neurons
H2 = 32
batch_size = 10 # every how many episodes we update the parameters
binary_input = False # True if we want to give the heaps as inputs represented in binary_input
episodes_for_training = 1 # Number of episodes used for training for each combination of tested parameters
decay_rate = 0.99   # decay factor for RMSProp leaky sum of grad^2
gamma = 0.99    # discount factor for reward
number_hidden_layers = 1 # Can only be 1 or 2
learning_rate = 1.0
opp_epsilon = 0.66 # The opponent will play opp_epsilon optimal
epsilon = 0.1 # Percentage of move the agent will take randomly

# Parameters for the training and testing
test_episode_nb = 100 # Number of episodes to run for test
episode_max = 50000 # Maximum number of episodes

# Initialize variables
heap = []
originalHeap = []
heapNb = 0
heapMax = 0
testing_results = []
testing_index = []

resume = False # resume from previous checkpoint
render = False

############### Function declarations####################

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
    # If not random, enter the board
    if not randomHeap:
        heapNb = raw_input("Enter number of heaps you want: ")
        heapNb = int(heapNb)
        for x in range(1,int(heapNb)+1):
            num = raw_input("Enter number of matches on heap %d: " % x)
            heap.append(int(num))
        heap = fillHeapZeros(heap)
        originalHeap = list(heap)

    # Random initialization
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

# Given an array of heaps, will return a vector representing the board with binary values
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

# Computes forward pass of the model with given input x
def policy_forward(x):
    # Convert heaps in binary
    if number_hidden_layers == 1:
        h1 = np.dot(model['W1'], x)
        h1[h1<0] = 0 # ReLU nonlinearity
        logp = np.dot(model['W2'].T, h1)
        p = sigmoid(logp)
        return p, h1 # return probability of taking action 2, and hidden state
    elif number_hidden_layers == 2:
        h1 = np.dot(model['W1'], x)
        h1[h1<0] = 0 # ReLU nonlinearity
        h2 = np.dot(model['W2'],h1)
        logp = np.dot(model['W3'].T, h2)
        p = sigmoid(logp)
        return p, h1, h2 # return probability of taking action 2, and hidden state

def policy_backward(eph1,eph2,epdlogp):
    """ backward pass. (eph1 is array of intermediate hidden states) """
    # dW2 = np.dot(eph.T, epdlogp).ravel()
    if number_hidden_layers == 1:
        dW2 = np.dot(eph1.T, epdlogp)
        dh = np.dot(epdlogp, model['W2'].T)
        dh[eph1 <= 0] = 0 # backpro prelu
        dW1 = np.dot(dh.T, epx)
        return {'W1':dW1, 'W2':dW2}
    elif number_hidden_layers == 2:
        dW3 = np.dot(eph2.T, epdlogp)
        dh2 = np.dot(epdlogp, model['W3'].T)
        dh2[eph2 <= 0] = 0 # backpro prelu
        dW2 = np.dot(dh2.T,eph1)
        dh1 = np.dot(dh2,model['W2'].T) # not sure
        dh1[eph1 <= 0] = 0 # backpro prelu
        dW1 = np.dot(dh1.T, epx)
        return {'W1':dW1, 'W2':dW2, 'W3':dW3}

############### Main program ####################
defineBoard()
# Adapt dimensions of input
if binary_input:
    D = max_heap_nb*3 #CHANGE # input dimensionality: number of heapNb (in binary)
else:
    D = max_heap_nb

# Define first turn randomly
agentTurn = bool(random.getrandbits(1)) # Bool which represents player's turn. 1 is agent, 0 is computer opponent

# Initialize randomly the network or reuse a previous one
if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    if number_hidden_layers == 1:
        model = {}
        model['W1'] = np.random.randn(H1,D) / np.sqrt(D) # "Xavier" initialization
        model['W2'] = np.random.randn(H1,max_heap_nb*max_heap_size) / np.sqrt(H1)
    elif number_hidden_layers == 2:
        model = {}
        model['W1'] = np.random.randn(H1,D) / np.sqrt(D) # "Xavier" initialization
        model['W2'] = np.random.randn(H1,H2) / np.sqrt(H2) # "Xavier" initialization
        model['W3'] = np.random.randn(H2,max_heap_nb*max_heap_size) / np.sqrt(H2)

grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory



xs,h1s,h2s,dlogps,drs = [],[],[],[],[]
running_reward = None
computerWin = False
playerWin = False
reward_sum = 0
episode_number = 0

while True:
    if render: printBoard(heap)
    if not agentTurn: # computer turn
        computerMove()
        heap = sortHeap(heap)
        agentTurn = True
        continue

    # Define input format
    if binary_input:
        x = heap_to_binary(heap)
    else:
        x = list(heap)

    reward = 0.0
    computerWin = isItEnd()
    if number_hidden_layers == 1:
        aprob, h1 = policy_forward(x)
    else:
        aprob, h1, h2 = policy_forward(x)

    xs.append(x) # Observation
    h1s.append(h1) # Hidden state
    if number_hidden_layers == 2:
        h2s.append(h2)

    if epsilon > random.uniform(0, 1): # random play (epsilon greedy)
        play = randomMove()
    else:   # Normal play
        # Search for chosen action
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
        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph1 = np.vstack(h1s)
        if number_hidden_layers == 2:
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
        if number_hidden_layers == 1:
            grad = policy_backward(eph1,0,epdlogp)
        elif number_hidden_layers == 2:
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

        # Test the model every test_episode_nb episodes
        if episode_number % test_episode_nb == 0:
            store_opp_epsilon = opp_epsilon # Store initial opp_epsilon value to restore it later
            opp_epsilon = 0.0
            # Store episode number associated to testing result
            testing_index.append(episode_number)
            win_number = 0.0
            for __ in range (test_episode_nb):
                computerWin = False
                playerWin = False
                heap = defineRandomBoard()
                heap = sortHeap(heap)
                agentTurn = bool(random.getrandbits(1)) #
                while True:
                    if not agentTurn: # computer turn
                        computerMove()
                        heap = sortHeap(heap)
                        agentTurn = True
                        continue

                    # Check if computer won previous turn
                    computerWin = isItEnd()
                    x = list(heap)
                    if number_hidden_layers == 1:
                        aprob,h1 = policy_forward(x)
                    elif number_hidden_layers == 2:
                        aprob,h1,h2 = policy_forward(x)
                    play = 0
                    for i in range(1,len(aprob)):  # Search biggest value in aprob for possible action
                        actionRemoveIndex = int(i)/int(max_heap_size)
                        actionRemoveNb = int(i)%int(max_heap_size)+1
                        if (heap[actionRemoveIndex] >= actionRemoveNb) and (aprob[i] > aprob[play]):
                            play = i

                    # Play greedy
                    if not computerWin:
                        actionRemoveIndex = int(play)/int(max_heap_size)
                        actionRemoveNb = int(play)%int(max_heap_size)+1
                        heap[actionRemoveIndex] -= actionRemoveNb
                        heap = sortHeap(heap)
                        playerWin = isItEnd()
                        agentTurn = False
                    done = playerWin or computerWin
                    if done:
                        if playerWin: # Store number of wins
                            win_number += 1.0
                        break;
            opp_epsilon = store_opp_epsilon # Restore opponent epsilon value
            testing_results.append(win_number)
        computerWin = False
        playerWin = False
        reward_sum = 0
        if randomHeap:
            heap = defineRandomBoard()
        else:
            heap = list(originalHeap)
        heap = sortHeap(heap)
        agentTurn = bool(random.getrandbits(1)) # Bool which represents player's turn. 1 is agent, 0 is computer opponent

    if reward != 0 and (episode_number % 1000 == 0) : # Nim has either +1 or -1 reward exactly when game ends.
        print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')

    # Stop at max episode defined
    if (episode_number == episode_max):
        break;

# Plots
plt.plot(testing_index, testing_results)
plt.title("Winning rate of the agent")
plt.xlabel("Run"); plt.ylabel("Games won [%]")
plt.axis([0, episode_max, 0, 105]); plt.grid(True)
# plt.show ## UNCOMMENT TO SHOW PLOT
plt.savefig('testing.pdf')
