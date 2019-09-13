### MDP Value Iteration and Policy Iteration
### Reference: https://web.stanford.edu/class/cs234/assignment1/index.html 
import numpy as np

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Evaluate the value function from a given policy.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS,nA]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """
    
    value_function = np.zeros(nS)
    ############################
    # YOUR IMPLEMENTATION HERE #
    # initialize value 
    V = value_function
    V_next = np.zeros(nS)

    isDone = False
    while not isDone:
        # iterate for each of the states
        for state in range(nS):
            value_next = 0
            # update value by adding up all possibilities in the state
            for action, p_policy in enumerate(policy[state]):
                for p_trans, nextstate, reward, terminal in P[state][action]:
                    value_next += p_policy * p_trans * (reward + gamma * V[nextstate])
            # log the value
            V_next[state] = value_next
        # check convergence and update value
        dist = 0
        for state in range(nS):
            # update distance
            dist = max(abs(V_next[state] - V[state]), dist)
            # update value function
            V[state] = V_next[state]
        isDone = (dist < tol)
    value_function = V
    ############################
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        A 2D array of floats. Each float is the probability of the action
        to take in that state according to the environment dynamics and the 
        given value function.
    """

    new_policy = np.ones([nS, nA]) / nA
	############################
	# YOUR IMPLEMENTATION HERE #
    V = value_from_policy
    Policy = new_policy
    for state in range(nS):
        # get the Q value for each of the state-action
        Q = np.zeros(nA)
        for action in range(nA):
            for p_trans, nextstate, reward, terminal in P[state][action]:
                    Q[action] += p_trans * (reward + gamma * V[nextstate])
        #select the best Q and set their value
        best_q = -1 # any value smaller than 0
        for action, q_value in enumerate(Q):
            if q_value > best_q:
                Policy[state] = np.zeros(nA)
                Policy[state][action] = 1
                best_q = q_value
            # ignore if q_value is not the best
    new_policy = Policy
	############################
    return new_policy


def policy_iteration(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: policy to be updated
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    new_policy: np.ndarray[nS,nA]
    V: np.ndarray[nS]
    """
    new_policy = policy.copy()
	############################
	# YOUR IMPLEMENTATION HERE #
    isDone = False
    while not isDone:
        last_policy = new_policy.copy()
        # evaluate the policy
        V = policy_evaluation(P, nS, nA, last_policy, gamma=0.9, tol=1e-8)
        # improve this policy
        new_policy = policy_improvement(P, nS, nA, V, gamma=0.9)
        # done if converge
        isDone = (abs(new_policy - last_policy) < tol).all()
    # calculate the value again in order to pass the test. but why?
    V = policy_evaluation(P, nS, nA, new_policy, gamma=0.9, tol=1e-8)   
	############################
    return new_policy, V

def value_iteration(P, nS, nA, V, gamma=0.9, tol=1e-8):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    V: value to be updated
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    policy_new: np.ndarray[nS,nA]
    V_new: np.ndarray[nS]
    """
    
    V_new = V.copy()
    ############################
    # YOUR IMPLEMENTATION HERE #
    isDone = False
    while not isDone:
        V_last = V_new.copy()
        for state in range(nS):
            # looks like the BV is exactly the same as searching for a best Q value
            Q = np.zeros(nA)
            for action in range(nA):
                for p_trans, nextstate, reward, terminal in P[state][action]:
                    Q[action] += p_trans * (reward + gamma * V_last[nextstate])
            V_new[state] = np.max(Q)
        # done if converge
        isDone = (abs(V_new - V_last) < tol).all()
    # extract optimal policy
    policy_new = policy_improvement(P, nS, nA, V_new, gamma=0.9)
    ############################
    return policy_new, V_new

def render_single(env, policy, render = False, n_episodes=100):
    """
    Given a game envrionemnt of gym package, play multiple episodes of the game.
    An episode is over when the returned value for "done" = True.
    At each step, pick an action and collect the reward and new state from the game.

    Parameters:
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as attributes.
    policy: np.array of shape [env.nS, env.nA]
      The action to take at a given state
    render: whether or not to render the game(it's slower to render the game)
    n_episodes: the number of episodes to play in the game. 
    Returns:
    ------
    total_rewards: the total number of rewards achieved in the game.
    """
    total_rewards = 0
    for _ in range(n_episodes):
        ob = env.reset() # initialize the episode
        done = False
        while not done:
            if render:
                env.render() # render the game
            ############################
            # YOUR IMPLEMENTATION HERE #
            # choose an action base on state and policy
            action = np.random.choice(env.nA, p=policy[ob])
            # choose the next state base on action and possibility
            possibilities = env.P[ob][action]
            p_trans = []
            nextstate = []
            reward = []
            terminal = []
            for i in range(len(possibilities)):
                p_trans.append(possibilities[i][0])
                nextstate.append(possibilities[i][1])
                reward.append(possibilities[i][2])
                terminal.append(possibilities[i][3])
            choice = np.random.choice(len(p_trans), p=p_trans)
            ob = nextstate[choice]
            # end if terminal
            done = terminal[choice]
        # accumulate reward
        total_rewards += reward[choice]
            
    return total_rewards



