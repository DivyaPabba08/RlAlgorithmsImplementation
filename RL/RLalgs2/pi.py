import numpy as np
from utils import action_evaluation

def policy_iteration(env, gamma, max_iteration, theta):
    """
    Implement Policy iteration algorithm.

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    P[state][action] is list of tuples. Each tuple contains probability, nextstate, reward, terminal
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
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
    """

    V = np.zeros(env.nS)
    policy = np.zeros(env.nS, dtype = np.int32)
    policy_stable = False
    numIterations = 0
    
    while  not policy_stable and numIterations < max_iteration:
        #Implement it with function policy_evaluation and policy_improvement
        ############################
        # YOUR CODE STARTS HERE
        V = policy_evaluation(env, policy, gamma, theta)
        policy, policy_stable = policy_improvement(env, V, policy , gamma)
        # YOUR CODE ENDS HERE
        ############################
        numIterations += 1
        
    return V, policy, numIterations


def policy_evaluation(env, policy, gamma, theta):
    """
    Evaluate the value function from a given policy.

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    
            env.nS: int
                    number of states
            env.nA: int
                    number of actions

    gamma: float
            Discount factor.
    policy: numpy.ndarray
            The policy to evaluate. Maps states to actions.
    theta: float
            The threshold of convergence.
    
    Outputs:
    V: numpy.ndarray
            The value function from the given policy.
    """
    ############################
    # YOUR CODE STARTS HERE
    V = np.zeros(env.nS)
    delta = float('inf')
    while theta <= delta:
        delta = 0
        for s in range(env.nS):
            qvalues=[]
            temp = V[s]
            a=policy[s]
            for k in range(len(env.P[s][a])):
                qvalues.append(env.P[s][a][k][0]*(env.P[s][a][k][2] + gamma*V[env.P[s][a][k][1]]))
            V[s]=sum(qvalues)    
            delta = max(delta,abs(V[s]-temp))
    # YOUR CODE ENDS HERE
    ############################

    return V





def policy_improvement(env, value_from_policy, policy, gamma):
    """
    Given the value function from policy, improve the policy.

    Inputs:
    env: OpenAI Gym environment
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

    value_from_policy: numpy.ndarray
            The value calculated from the policy
    policy: numpy.ndarray
            The previous policy.
    gamma: float
            Discount factor.

    Outputs:
    new policy: numpy.ndarray
            An array of integers. Each integer is the optimal action to take
            in that state according to the environment dynamics and the
            given value function.
    policy_stable: boolean
            True if the "optimal" policy is found, otherwise false
    """
    ############################
    # YOUR CODE STARTS HERE
    policy_stable = True
    for s in range(env.nS):
        temp = policy[s]
        qvalue=[]
        qvalues_total=[]
        qvalue_sum=0
        for a in range(env.nA):
            qvalue = [env.P[s][a][i][0]*(env.P[s][a][i][2] + gamma*value_from_policy[env.P[s][a][i][1]]) 
            for i in range(len(env.P[s][a]))]
            qvalue_sum=sum(qvalue)
            qvalues_total.append(qvalue_sum)   
        policy_new = np.argmax(qvalues_total) 
        if temp!= policy_new:
            policy[s] = policy_new
            policy_stable = False
    # YOUR CODE ENDS HERE
    ############################

    return policy, policy_stable