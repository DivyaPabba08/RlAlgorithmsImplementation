import numpy as np
from utils import action_evaluation

def value_iteration(env, gamma, max_iteration, theta):
    """
    Implement value iteration algorithm. 

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    the transition probabilities of the environment
                    P[state][action] is list of tuples. Each tuple contains probability, nextstate, reward, terminal
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    gamma: float
            Discount factor.
    max_iteration: int
            The maximum number of iterations to run before stopping.
    theta: float
            The threshold of convergence.
    
    Outputs:
    V: numpy.ndarray
    policy: numpy.ndarray
    numIterations: int
            Number of iterations
    """

    V = np.zeros(env.nS)
    numIterations = 0

    #Implement the loop part here
    ############################
    # YOUR CODE STARTS HERE
    delta = float('inf')
    while theta <= delta or numIterations < max_iteration:
        delta = 0
        for s in range(env.nS):
            temp = V[s]
            qvalue=[]
            qvalues_total=[]
            qvalue_sum=0
            for a in range(env.nA):
                qvalue = [env.P[s][a][i][0]*(env.P[s][a][i][2] + gamma*V[env.P[s][a][i][1]]) for i in range(len(env.P[s][a]))]
                qvalue_sum=sum(qvalue)
                qvalues_total.append(qvalue_sum)
            V[s] = max(qvalues_total)
            delta = max(delta,abs(V[s]-temp))
        numIterations += 1 

    # YOUR CODE ENDS HERE
    ############################
    
    #Extract the "optimal" policy from the value function
    policy = extract_policy(env, V, gamma)
    
    return V, policy, numIterations

def extract_policy(env, v, gamma):

    """ 
    Extract the optimal policy given the optimal value-function.

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    P[state][action] is tuples with (probability, nextstate, reward, terminal)
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    v: numpy.ndarray
        value function
    gamma: float
        Discount factor. Number in range [0, 1)
    
    Outputs:
    policy: numpy.ndarray
    """

    policy = np.zeros(env.nS, dtype = np.int32)
    ############################
    # YOUR CODE STARTS HERE
    for s in range(env.nS):
        qvalues=[]
        qvalue=[]
        for a in range(env.nA):
            for k in range(len(env.P[s][a])):
                qvalues.append(env.P[s][a][k][0]*(env.P[s][a][k][2] + gamma*v[env.P[s][a][k][1]]))
            qvalue.append(sum(qvalues))
            policy[s] = np.argmax(qvalue)
    # YOUR CODE ENDS HERE
    ############################

    return policy