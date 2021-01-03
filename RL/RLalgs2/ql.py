import numpy as np
from utils import epsilon_greedy
import random

def QLearning(env, num_episodes, gamma, lr, e):
    """
    Implement the Q-learning algorithm following the epsilon-greedy exploration.

    Inputs:
    env: OpenAI Gym environment 
            env.P: dictionary
                    P[state][action] are tuples of tuples tuples with (probability, nextstate, reward, terminal)
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    num_episodes: int
            Number of episodes of training
    gamma: float
            Discount factor.
    lr: float
            Learning rate.
    e: float
            Epsilon value used in the epsilon-greedy method.

    Outputs:
    Q: numpy.ndarray
    """

    Q = np.zeros((env.nS, env.nA))
    
    #TIPS: Call function epsilon_greedy without setting the seed
    #      Choose the first state of each episode randomly fosr exploration.
    ############################
    # YOUR CODE STARTS HERE
    for i in range(num_episodes):
        state=np.random.randint(0,env.nS)
        statelist=[]
        for j in range(0,env.nS):
            if j == state:
                statelist.append(1)
            else:
                statelist.append(0)
        env.isd=statelist
        env.reset()
        terminal = False
        while not terminal:
            action=epsilon_greedy(Q[state],e)
            next_state, reward, terminal, prob = env.step(action)
            Q[state, action]=Q[state,action]+lr*(reward+gamma*np.max(Q[next_state]))-lr*Q[state,action]
            state = next_state   
            
    # YOUR CODE ENDS HERE
    ############################

    return Q