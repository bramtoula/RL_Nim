# Reinforcement learning for Nim

## Introduction

> We want to compare the learning of Nim with 3 different reinforcement learning methods: Q-Learning, Deep Q-Learning, and Deep Policy Gradient.

## Installation

> The code is written in **Python2**.

> It requires the following packages: "matplotlib", "numpy", and "jupyter". An update of pip might be necessary: "*pip install --upgrade pip*"

> To install the packages, type in a terminal "*pip install package_name*"

>**NOTA BENE:** With windows all those instructions have to be preceded by "*python -m *"

> It also requires "pytorch". You can go to http://pytorch.org/ for instructions. The installation command is specific to your machine!

## Run the code

> The 3 implementations are in different files.

> To run the **Q-Learning**:
- Open a terminal and go where the jupyter notebooks (.ipynb) are.
- Type "*jupyter notebook*". After a while a new tab should open in your navigator.
- From there you can open the two notebooks:
    - *QL_Nim.ipynb* : The main code that make an agent learn to play nim, then evaluate it, and finally propose to play against it
    - *QL_gridSearch.ipynb* : The code that performs the grid search over the parameters to find the optimal combination
- Inside a notebook you can run the code block by block by cliking on it and then pressing "*Ctrl*"+"*Enter*". The output appears at the bottom of the block.

> To run the **Deep Q-Learning**:
- Simply run "*python DQN_Nim.py*" from a terminal. The script will perform a parameter grid search over the method. You can adjust parameters at the top of the script.

> To run the **Deep Policy Gradient**:
- From a terminal, run:
    - "*python PG_Nim.py*" : Launches a script that trains an agent for a while, and then plots its winning rate during the learning. Inside the file, the parameters between the lines "###### CAN CHANGE #####" may be manually tuned.
    - "*python PG_gridSearch*" : Launches a script that performs a grid search. The agent trains for a while, then the optimality measure will be plotted depending on his learning rate, his exploratory term, and the opponent's randomness. Inside the file, the parameters between the lines "###### CAN CHANGE #####" may be manually tuned.