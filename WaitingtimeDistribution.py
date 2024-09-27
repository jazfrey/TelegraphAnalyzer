# -------------- Contents of Libary --------------------------------
# - Tool for fitting timestampdata to a markov modell via the waitingtime distribution for  singlephoton measurements

import numpy as np
from scipy.linalg import expm
from scipy.linalg import null_space
from numba import njit, objmode, prange
from lmfit import Minimizer

# Handles debug outputs
DEBUG = True


def eigenvector_for_eigenvalue_zero(A):
    """ 
    Calculates the eigenvalue to the eigenvalue 0, because that is the statioinary state 
    """
    # Calculate the null space of A: det(A - 0*I) = 0
    null_space_A = null_space(A)

    if null_space_A.size == 0:
        print(f"No eigenvectors found for the eigenvalue 0 of matrix {A}!")
        return None  # No eigenvector corresponding to eigenvalue 0
    else:
        # Return the first basis vector of the null space
        return null_space_A[:, 0]
    

def rates_to_matrix(rates, subDiag=True ):
    """
    Convert a dictionary of rates to a continuous-time Markov process transition rate matrix, that is already in the right form to be exponentiated to get the propagator, but without the transitions given in the jump-array

    Parameters
    ----------
    rates : dict
        Dictionary containing the rates for each transition. Keys should be in the format
        'from_state->to_state' and values should be non-negative floats representing the rates.

    subDiag: bool
        Enables the subtraction of the diagonal
        
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
    if subDiag:
        for i in range(n):
            matrix[i, i] = -sum(matrix[i])

    return matrix.T

@njit(parallel=True, fastmath=True)
def TraceProbability(data, L, L0, j1, j2, dtau, doParallel=True):
    """ Calculates the whole probability of a trace by calculating the probability of the waiting time. FIXME This method is only coded for the case of the singlePhoton data given in the 'Making every photon count: A quantum polyspectra approach to the dynamics of
    blinking quantum emitters at low photon rates without binning'-Paper, where always the same jump from 1->2 happens. But a general implementation could be written for diffrent state transitions. """
    
    p_trace = 0

    # calculating the steady eigenstate
    with objmode(rho_0='float64[:]'):
        rho_0 = eigenvector_for_eigenvalue_zero(L)
    
    # Precalculation of the scalar product
    firstJump = np.dot(j1, np.ascontiguousarray(rho_0))
    # normalization-Factor
    I = np.sum( firstJump )

    if doParallel:
        # parallelized loop
        #print("Parallelized")
        for i in prange( 0, len(data)-1 ):
            waitingtime = data[i+1] - data[i]

            # Time evolution during the waitingtime
            with objmode(matExp='float64[:,:]'):
                matExp = np.ascontiguousarray(expm(L0*waitingtime))

            # calculate the waitingtime-probability-distribution 
            w_tau = np.sum( np.dot( j2, np.dot(np.ascontiguousarray(matExp), firstJump)) ) / I

            # check if the probability is too small resulting in 0 and therefore crashing np.log
            if w_tau != 0:
                # Calculate Probability by multiplying
                p_trace += np.log( w_tau*dtau )
            else:
                print("prob = 0")
    else:
        #print("Unparallelized")
        # Strange behaviour, the njit(parallel=True) option parallelizes this part of code but not the from above
        for i in range( 0, len(data)-1 ):
            waitingtime = data[i+1] - data[i]

            # Time evolution during the waitingtime
            with objmode(matExp='float64[:,:]'):
                matExp = expm(L0*waitingtime)

            # calculate the waitingtime-probability-distribution 
            w_tau = np.sum( np.dot( j2, np.dot(np.ascontiguousarray(matExp), firstJump)) ) / I

            # check if the probability is too small resulting in 0 and therefore crashing np.log
            if w_tau != 0:
                # Calculate Probability by multiplying
                p_trace += np.log( w_tau *dtau )
            else:
                print("prob = 0")

    return p_trace


class System():
    """"
    Class that holds all the informations about the markov system needed for a fit over the waitingtime distribution
    """
    def __init__(self, rates, jumps):
        # Initialisation of L
        self.L = rates_to_matrix(rates)
        self.jumps_matrix = [ rates_to_matrix(jump, False) for jump in jumps]

        # Calculation of L0
        self.L0 = np.array(self.L)
        for jump_matrix in self.jumps_matrix:
            self.L0 -= jump_matrix


class FitSystem():

    def __init__( self, set_system, path, doParallel, photonRatio):
        self.data = np.load(path)
        self.set_system = set_system
        self.doParallel = doParallel
        self.photonRatio = photonRatio

    def fit_timestamps(self, params):
        """
        Starting the minimizing routing, that calls the objective-function that returns the probability of a given trace for the given set of parameters
        """
        self.paramCount = len( params.valuesdict().values() )

        fitter = Minimizer(self.objective, params=params)
        result = fitter.minimize(method="leastsq", ftol=1e-7, xtol=1e-7) #FIXME das hier ist noch random
        return result
    
    def objective(self, params):
        system = self.set_system(params, self.photonRatio)

        # Hier nochmal über den Vorfaktor überlegen. 
        #norm = 100000 * np.max( list(params.valuesdict().values()) )**-1 # FIXME hier könnte auch noch besser gewählt werden. Maximum von gamma_up, gamma_down
        norm = 10 / 750
        p_trace = TraceProbability(data=self.data, L=system.L, L0=system.L0, j1=np.ascontiguousarray(system.jumps_matrix[0]), j2=np.ascontiguousarray(system.jumps_matrix[1]), dtau=norm, doParallel=self.doParallel)

        if DEBUG:
            print("In:", params["gamma_in"].value)
            print("Out:", params["gamma_out"].value)
            print("Ph:", params["gamma_ph"].value)
            print("Det:", params["gamma_det"].value)
            print("ln(p_trace):", p_trace, "\n")

        out = np.full(self.paramCount, p_trace**2)
        return out