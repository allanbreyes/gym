OpenAI Gym Solutions
====================

## Requirements

- Python v2.7.13
- virtualenv
- pip

## How to Run

1. Run `virtualenv venv`
2. Run `source venv/bin/activate`
3. Run `pip install -r requirements.txt`

### Experiments

#### Solving Markov Decision Processes

In this experiment, convergence and performance of value iteration and policy
iteration are compared for 3 different MDPs, including:

1. `FrozenLake-v0`
2. `FrozenLake8x8-v0`
3. `Taxi-v2`

Reproduce the results by running `python analysis/mdp.py`

#### Q-learning Agents

A Q-learner reinforcement learning algorithm was applied to the "Toy Text"
environments. You can reproduce the results by running:

- `python frozen_lake/q_learning.py`
- `python taxi/q_learning.py`
