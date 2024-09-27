# ------------- Contents of Libary ----------------------------------------------------------
# - Tool to gernerate transition matrices form any transition dictionary
# - Tool to sample data form any markov system characterized by the transition dictionary
# - Forward algorithm that determines the probabiltity of a noisy data trace and then fits the best parameters

import numpy as np
from scipy.linalg import expm
from numba import njit, objmode
from lmfit import Minimizer

# Handles debug outputs
DEBUG = False

def rates_to_matrix(rates):
    """
    Convert a dictionary of rates to a continuous-time Markov process transition rate matrix, that is already in the right form to be exponentiated to get the propagator.

    Parameters
    ----------
    rates : dict
        Dictionary containing the rates for each transition. Keys should be in the format
        'from_state->to_state' and values should be non-negative floats representing the rates.

    Returns
    -------
    np.ndarray
        A NumPy array representing the transition rate matrix.

    Raises
    ------
    ValueError
        If the keys in the dictionary are not in the 'from_state->to_state' format or if rates are negative.

    Example
    -------
    >>> rates = {"1->2": 0.5, "2->1": 0.7, "2->3": 0.2}
    >>> rates_to_matrix(rates)
    array( [[-0.5  0.7  0. ]
            [ 0.5 -0.9  0. ]
            [ 0.   0.2 -0. ]] )
    """

    # Validate input and identify unique states
    states = set()
    for key, rate in rates.items():
        try:
            from_state, to_state = key.split("->")
            rate = float(rate)
        except ValueError:
            raise ValueError(
                "Invalid key or rate value. Keys should be 'from_state->to_state' and rates should be non-negative numbers.")

        if rate < 0:
            raise ValueError("Rates should be non-negative numbers.")

        states.add(from_state)
        states.add(to_state)

    states = sorted(list(states))
    n = len(states)

    # Initialize a zero matrix
    matrix = np.zeros((n, n))

    # Fill the transition rates
    for key, rate in rates.items():
        from_state, to_state = key.split("->")
        i, j = states.index(from_state), states.index(to_state)
        matrix[i, j] = rate

    # Fill the diagonal elements such that each row sums to zero
    for i in range(n):
        matrix[i, i] = -sum(matrix[i])

    return matrix.T

@njit
def add_whitenoise(path, Amplitude):
    """
    Adds whitenoise to any given data array
    """
    with objmode(measurement='float64[:]'):
        measurement = np.load(path)
    noise = Amplitude * np.random.standard_normal(len(measurement))
    result = measurement + noise
    return result

@njit
def getPropagator(transitionMatrix, dt):
    """
    Calculates the timeindependent propagator, that solves the SME  
    """
    with objmode(G='float64[:,:]'):
        G = expm( transitionMatrix*dt)
    return G

@njit
def simulate_markov_chain(cumulative_P, initial_state, num_steps):
    """ 
    Simulate the Markov chain using Numba for acceleration.
    """
    states = np.empty(num_steps + 1, dtype=np.int32)  # Pre-allocate array for speed
    states[0] = initial_state
    current_state = initial_state

    #print(cumulative_P[:, current_state])

    for i in range(1, num_steps + 1):
        random_value = np.random.random()  # Generate a single random number
        # Find the next state using searchsorted: finds the state that is closest to the random number => rolling a dice with the probability distribution of the propagtor
        current_state = np.searchsorted(cumulative_P[:, current_state], random_value, side="right")
        states[i] = current_state

    return states

@njit
def gauss(x, sigma, mu):
    """
    Gaussian standarddistribution with peak mu and standarddeviation sigma
    """
    result = (1 / (np.sqrt(2*np.pi) * sigma) ) * np.exp( -0.5 * ((mu - x) / (sigma))**2 )
    return result

@njit
def NoisyTraceProbability(G, data, measurement_op, initial_state, sigma):
    """
    Calculate the Probability of a noisy-trace by guessing the state by the gaussian distribution. 
    """
    p_trace = log_p_trace = 0
    state_count = len(measurement_op)
    next_state = np.empty(state_count)
    p_distribution = np.empty(state_count)

    # First Probability vector
    current_state = np.zeros(state_count)
    current_state[initial_state] = 1

    for i in range(0, len(data)-1):
        # Propagating the current state u -> v
        next_state = np.dot(G, current_state)
        # Calculating the probability of the data point for belonging to a certain state
        p_distribution = gauss(x=data[i], sigma=sigma, mu=measurement_op)
        # Normalization of the distribution result's 
        p_distribution = p_distribution / np.sum(p_distribution)
        # Probability of the whole jump
        p_trace = np.dot(p_distribution, next_state)
        log_p_trace += np.log(p_trace)

        # Calculating the next probability vector
        current_state = (p_distribution * next_state) / p_trace 
        
    return log_p_trace


class SimSystem:
    """ 
    Class that stores the information that characterizes the markov system
    """

    def __init__( self, rates_dict, measurement_op, sigma = 0, initial_state = 0 ):
        self.rates_dict = rates_dict
        self.transition_matrix = rates_to_matrix(rates_dict)
        self.measurement_op = measurement_op            # Array that assigns every state it's measurement value
        self.sigma = sigma
        self.initial_state = initial_state # initial_state = 0 means that the first entry (array index 0) get's the probability of 1


    def simulate_discrete_trace(self, total_time, sampling_rate):
        """
        This method generates a sampled measurement signal from any markov system described by the transition matrix 
        """
        num_steps = int(total_time * sampling_rate)

        # Create the discrete-time transition matrix
        P = getPropagator(self.transition_matrix, 1 / sampling_rate)
        # print(f" Propagator P \n {P}")
        cumulative_P = np.cumsum(P, axis=0)             # Adds up all the entries of the matrix in a row to build a list in which, the length of the distances match the probabilities to allow for randomly valuepicking

        # Normalize all rows to get the intervals that are distributed with the right probability along [0, 1]
        for i in range( 0, np.shape(cumulative_P)[1] ):
            cumulative_P[:, i] = cumulative_P[:, i] / np.max(cumulative_P[:, i])
        # print(f" Cumulative P \n {cumulative_P}")

        # Simulate the Markov chain
        states = simulate_markov_chain(cumulative_P, self.initial_state, num_steps)

        measurement = np.zeros(len(states))
        for i, state in enumerate(states):
            measurement[i] = self.measurement_op[state]

        return states, measurement


class FitSystem():
    """
    Class that stores the information needed for fitment and fitting routine
    """

    def __init__( self, set_system, datapath, sampling_rate):
        self.set_system = set_system
        self.sampling_rate = sampling_rate
        self.data = np.load(datapath)


    def fit_sampled_noisy_data(self, params):
        """
        Given params (a Parameter()-Class object) and a rate dictionary with the transitions and parameter names as value, this finds the parameter that result in the biggest trace probability.
        For this the Minimizer-Class calls the objectiveFitNoiseSamples()-Function that returns the probability: log(p_trace)
        """
        # Count the number of parameters, to determine the dimensionality of the output array (needed for the "least_sq" method )
        self.paramCount = len( params.valuesdict().values() )

        # Minimizer is a routine that finds the value closest to 0. Works here for minimization because for values smaller than 1, following holds true x -> 1: ln(x) -> 0 
        fitter = Minimizer(self.objectiveFitNoiseSamples, params=params)
        result = fitter.minimize(method="leastsq", ftol=1e-8)

        return result
    
    def objectiveFitNoiseSamples(self, params):
        """"
        Calculates the probability of a noisy Trace, has the right syntax to be called by the Minimizer()-Wrapper
        """
        system = self.set_system(params)

        # Calculating the time evolution propagator G(dt)
        G = getPropagator(transitionMatrix=system.transition_matrix, dt= 1 / self.sampling_rate)
        
        p_trace = NoisyTraceProbability(G, self.data, system.measurement_op, system.initial_state, system.sigma)
        out = np.full(self.paramCount, p_trace)

        if DEBUG:
            print("Sigma:", params["sigma"].value)
            print("In:", params["gamma_12"].value)
            print("Out:", params["gamma_21"].value)
            print("n1:", params["n1"].value)
            print("n2:", params["n2"].value)
            print("ln(p_trace):", p_trace, "\n")

        return out