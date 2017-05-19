# RL_Nim
##Python##

The code is written in **Python2**

## Temporal-Difference Learning
##Installation##

a) The code requires the packages "matplotlib" and "numpy" to be installed. If needed, run "*pip install matplotlib*" and "*pip install     numpy*" (it may be possible that pip needs upgrading. If so, run "*pip install --upgrade pip*" beforehand). 

b) To run the scripts, you will need jupyter notebook. To install it, run "*pip install jupyter*"

**NOTA BENE**: On windows the commands will rather be like this: "*python -m pip install package_name*"

##Running TD-learning##

a) Once everything is installed, open a terminal and go in the folder where the jupyter notebook (.ipynb) are. Then type "*jupyter notebook*" to launch the notebook (a new page on your internet navigator will open after a while). From there you will be able to select a notebook to open.

b) There are two notebook: "TD_main.ipynb" which contains the main code for the learning (you'll be able to make an agent learn, then do some plots to evaluate it, and even play against it), and "TD_gridSearch.ipynb" which contains the code needed to perform the grid search on the parameters.

c) Once inside the notebook, you can run code-block by code-block the code by clicking in the block and pressing "Ctrl"+"Enter"

## Deep Reinforcement Learning
Results for Deep Q-Learning and Deep Policy Gradient are not as convincing as Q-Learning. However, you can run our implementations by following the instructions below:

##Running Deep Q-Learning##

a) Make sure you have pytorch installed. You can go to http://pytorch.org/ for instructions. The installation command is specific to your machine!

b) You will also need run "pip install matplotlib" if you do not already have the package installed.

c) run "python DQN_Nim.py", the script will perform a parameter grid search over the method.
    You can adjust parameters at the top of the script.
    You can plot the output of the grid search by uncommenting the code at the bottom of the script.

## Some sources:
Demo of RL using Deep Q-Learning: http://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html

Website of Richard S. Sutton (RL book & courses): http://incompleteideas.net/sutton/

An Introduction to RL (Online textbook): http://incompleteideas.net/sutton/book/the-book.html

Courses on RL (Sutton): http://incompleteideas.net/sutton/609%20dropbox/

Reinforcement Learning for Board Games: The Temporal Difference Algorithm: http://www.gm.fh-koeln.de/ciopwebpub/Kone15c.d/TR-TDgame_EN.pdf

RL for NIM: http://www.diva-portal.org/smash/get/diva2:814832/FULLTEXT01.pdf

RL for NIM 2:http://www.csc.kth.se/utbildning/kth/kurser/DD143X/dkand11/Group6Lars/erikjarleberg.pdf
