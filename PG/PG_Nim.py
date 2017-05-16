""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import cPickle as pickle
import gym
import random
import os
import platform
from time import sleep


# hyperparameters
max_heap_nb = 5
max_heap_size = 5
H = 30 # number of hidden layer neurons # CHANGE
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False
binary_input = False # True if we want to give the heaps as inputs represented in binary_input
epsilon = 0.9 # The opponent will play epsilon optimal

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
    os.system('clear')
    print "Let's start by defining our game:"
    heapNb = raw_input("Enter number of heapNb you want: ")
    heapNb = int(heapNb)
    for x in range(1,int(heapNb)+1):
        num = raw_input("Enter number of matches on heap %d: " % x)
        heap.append(int(num))
    for x in range(int(heapNb)+1,max_heap_nb+1):
        heap.append(0)
    heap = np.sort(heap)
    heap[:] = heap[::-1]
    originalHeap = list(heap)
    heapMax = max(heap)



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
    if epsilon > random.uniform(0, 1): # random move
        heap[np.argmax(heap)]-=random.randint(1,max(heap))
    else:
        if nimSum() == 0: # optimal move
            heap[np.argmax(heap)]-=random.randint(1,max(heap))
        else:
            heap[winningHeap()]^=nimSum()



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
    model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
    model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
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
    h = np.dot(model['W1'], x)
    h[h<0] = 0 # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0 # backpro prelu
    dW1 = np.dot(dh.T, epx)

    return {'W1':dW1, 'W2':dW2}

xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
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
        heap = np.sort(heap)
        heap[:] = heap[::-1]
        agentTurn = True
        continue

    if binary_input:
        x = heap_to_binary(heap)
    else:
        x = list(heap)

    computerWin = isItEnd()
    aprob, h = policy_forward(x)
    xs.append(x) # observation
    hs.append(h) # hidden state
    reward = 0.0
    finish = False
    actionsIndexNb = np.sum(heap)

    play = int(actionsIndexNb*aprob)+1

    if actionsIndexNb != 0:
        y = (play-0.5)/actionsIndexNb
    else:
        y = 0

    dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
    if not computerWin:
        # Player move
        for i in range (0,len(heap)):
            temp = play - sum(heap[0:i+1])
            if temp <= 0:
                actionRemoveIndex = i
                actionRemoveNb = int(play - sum(heap[0:i]))
                finish = True
                break
        # actionRemoveIndex = int(play)/heapMax
        # actionRemoveNb = int(play)%heapMax
        # if heap[actionRemoveIndex] >= actionRemoveNb:
        #     finish = True
        #     heap[actionRemoveIndex] -= actionRemoveNb # CHANGE METHOD TO ACQUIRE MOVE ; USE 2 output NEURONS ? VERIFY IF POSSIBLE MOVE ?
        # else : # random play but penalize
        #     if max(heap) == 1:
        #       heap[heap.index(max(heap))]-= 1
        #     if max(heap) > 1:
        #       heap[heap.index(max(heap))]-=random.randint(1,max(heap)) # Change to total random play
        #     reward = -0.5
        #     finish = True
        heap[actionRemoveIndex] -= actionRemoveNb
        heap = np.sort(heap)
        heap[:] = heap[::-1] #Sort descending
        playerWin = isItEnd()
        agentTurn = False

        # Should be in main loop ?
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
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs,hs,dlogps,drs = [],[],[],[] # reset array memory


        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)
        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epdlogp)
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
        heap = list(originalHeap)
        agentTurn = bool(random.getrandbits(1)) # Bool which represents player's turn. 1 is agent, 0 is computer opponent

    if reward != 0 and (episode_number % 1000 == 0) : # Pong has either +1 or -1 reward exactly when game ends.
        print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')
