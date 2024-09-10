"""The Black Merton Scholes economy

Local implementation of some useful functions for the BMS model. 
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
import unittest

# Local packages
#from .jupyter_notebook import *

def d1(S, K, r, y, T, sigma): 
    '''Calculate d1 from the Black, Merton and Scholes formula'''
    return (np.log(S/K) + (r - y + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

def d2(S, K, r, y, T, sigma): 
    '''Calculate d2 from the Black, Merton and Scholes formula'''
    return (np.log(S/K) + (r - y - 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

def delta(S, K, r, y, T, sigma, is_call):  
    '''Return Black, Merton, Scholes delta of the European (call, put)'''
    _d1 = d1(S, K, r, y, T, sigma)    
    d_sign = np.where(is_call, 1, -1)
    return d_sign*norm.cdf(d_sign*_d1)

def gamma(S, K, r, y, T, sigma, is_call=None):
    '''Return Black, Merton, Scholes gamma of the European call or put

    Accepts is_call argument for consistency in the functions' signatures, but it is neglected
    '''
    _d1 = d1(S, K, r, y, T, sigma)
    return np.exp(-y*T)*norm.pdf(_d1)/(S*sigma*np.sqrt(T))

def theta(S, K, r, y, T, sigma, is_call):
    _d1 = d1(S, K, r, y, T, sigma)
    _d2 = _d1 - sigma*np.sqrt(T)
    d_sign = np.where(is_call, 1, -1)
    return -np.exp(-y*T)*S*norm.pdf(_d1)*sigma / (2*np.sqrt(T)) \
        -d_sign * r*K*np.exp(-r*T)*norm.cdf(d_sign*_d2)  +  d_sign * y*S*np.exp(-y*T)*norm.cdf(d_sign*_d1)

def vega(S, K, r, y, T, sigma, is_call=None):
    '''Return Black, Merton, Scholes vega of the European call or put

    Accepts is_call argument for consistency in the functions' signatures, but it is neglected
    '''
    _d1 = d1(S, K, r, y, T, sigma)
    return S*np.exp(-y*T)*norm.pdf(_d1)*np.sqrt(T)

def vanna(S, K, r, y, T, sigma, is_call=None):
    '''Return Black, Merton, Scholes vanna of the European call or put

    Accepts is_call argument for consistency in the functions' signatures, but it is neglected
    '''
    _d1 = d1(S, K, r, y, T, sigma)
    _d2 = _d1 - sigma*np.sqrt(T)    
    return -np.exp(-y*T)*norm.pdf(_d1)*(_d2/sigma)


def option_price(S, K, r, y, T, sigma, is_call, ret_delta=False):  
    '''Return Black, Merton, Scholes price of the European option'''
    _d1 = d1(S, K, r, y, T, sigma)
    _d2 = _d1 - sigma*np.sqrt(T)
    
    # d_sign: Sign of the the option's delta
    d_sign = np.where(is_call, 1, -1)
    delta = d_sign*norm.cdf(d_sign*_d1)
    premium = np.exp(-y*T)*S*delta - d_sign*np.exp(-r*T)*K*norm.cdf(d_sign*_d2);
    if ret_delta:
        return premium, delta
    return premium
    
def _implied_volatility(opt_price, S, K, r, y, T, is_call, init_vol=0.6):
    '''Inverse the BMS formula numerically to find the implied volatility'''
    def pricing_error(sig):
        sig = abs(sig)
        return option_price(S,K,r,y,T,sig,is_call) - opt_price
    return fsolve(pricing_error, init_vol)
implied_volatility = np.vectorize(_implied_volatility)

def simulate_underlying(S0, r, y, sigma, dt, shocks):
    '''Simulate the GMB based on the user-provided standard normal shocks

    Parameters:
        S, r, y, sigma, : as usual
        dt     : the time step length in the simulation
        shocks : A (n_steps x n_sim) matrix of standard Normal shocks for a 
                 simulation with n_steps time steps and n_sim paths

    Returns:
        S : A (n_steps+1 x n_sim) matrix with n_sim paths of the underlying simulated over n_steps time steps, 
            starting at time 0
    '''
    n_steps, n_sim = shocks.shape
    S = np.empty((n_steps+1,n_sim))
    S[0,:] = S0;
    for tn in range(n_steps):
        S[tn+1,:] = S[tn,:]*np.exp( (r-y-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*shocks[tn,:] )
    return S

def delta_hedge(premium, S_t, K, r, y, T, sigma, dt, is_call):
    """Delta hedges a **short** position in the option across simulated paths S_t

    S_t must start with a row including the current value of the stock
    """
    raise RuntimeError('Implement for Assignment 1')
    
   
class TestModule(unittest.TestCase):
    """Simple test class for a single set of BMS inputs.

    This class tests the various functions in this module that calculate greeks and option
    price based on the BMS model, against values obtained with otherwise languages & packages

    Attributes:
        S (float): The value of the underlying.
        K (float): The strike price.
        r (float): The risk-free rate.
        y (float): The dividend yield.
        T (float): The time to expiration of the option.
        sigma (float): The volatility of the underlying (static given BMS).

    Methods:
        setUp(): Preparing the necessary setup for each test.
        test_d1(): Test the calculation of d1.
        test_d2(): Test the calculation of d2.
        test_delta(): Test the calculation of delta.
        test_gamma(): Test the calculation of gamma.
        test_theta(): Test the calculation of theta.
        test_vega(): Test the calculation of vega.
        test_vanna(): Test the calculation of vanna.
        test_option_price(): Test the calculation of the option price.
        test_implied_volatility(): Test the calculation of implied volatility.
        test_simulate_underlying(): Test the simulate_underlying function.

    """

    # https://docs.python.org/3/library/unittest.html#unittest.TestCase.setUp
    def setUp(self):
        """Preparing the necessary setup for each test.

        This method is executed before each test case to set up the required environment.
        It initializes the values of the underlying, strike price, risk-free rate, dividend yield,
        time to expiration, and volatility.
        """
        self.S = 100.0
        self.K = 95.0
        self.r = 0.05
        self.y = 0.02
        self.T = 1
        self.sigma = 0.2

    def test_d1(self):
        """Tests the calculation of d1.

        This test verifies the correctness of the d1 calculation by comparing the calculated value
        to the expected value using self.assertAlmostEqual(). 
        """
        calculated_d1 = d1(self.S, self.K, self.r, self.y, self.T, self.sigma)
        self.assertAlmostEqual(0.5064664719377524, calculated_d1, places=3)

    def test_d2(self):
        """Tests the calculation of d2.

        This test validates the correctness of the d2 calculation by comparing the calculated value
        to the expected value using self.assertAlmostEqual().
        """
        calculated_d2 = d2(self.S, self.K, self.r, self.y, self.T, self.sigma)
        self.assertAlmostEqual(0.30646647193775234, calculated_d2, places=3)

    def test_delta(self):
        """Tests the calculation of delta.

        This test function verifies the correctness of the delta calculation by comparing the calculated delta
        to the expected value using self.assertAlmostEqual(). Equivalence was not found with R nor Matlab so
        we are using Python function.
        """
        calculated_delta = delta(self.S, self.K, self.r, self.y, self.T, self.sigma, is_call=True)
        self.assertAlmostEqual(0.6937353895396909, calculated_delta, places=8)

    def test_gamma(self):
        """Tests the calculation of gamma.

        This test function checks the correctness of the gamma calculation by comparing the calculated gamma
        to the expected value using self.assertAlmostEqual().
        """
        calculated_gamma = gamma(self.S, self.K, self.r, self.y, self.T, self.sigma)
        self.assertAlmostEqual(0.0171986403, calculated_gamma, places=8)

    def test_theta(self):
        """Tests the calculation of theta.

        This test function validates the correctness of the theta calculation by comparing the calculated theta
        to the expected value using self.assertAlmostEqual().
        """
        calculated_theta = theta(self.S, self.K, self.r, self.y, self.T, self.sigma, is_call=True)
        self.assertAlmostEqual(-4.882797197, calculated_theta, places=8)

    def test_vega(self):
        """Tests the calculation of vega.

        This test function verifies the correctness of the vega calculation by comparing the calculated vega
        to the expected value using self.assertAlmostEqual().
        """
        calculated_vega = vega(self.S, self.K, self.r, self.y, self.T, self.sigma)
        self.assertAlmostEqual(34.39728061, calculated_vega, places=8)

    def test_vanna(self):
        """Tests the calculation of vanna.

        This test function validates the correctness of the vanna calculation by comparing the calculated vanna
        to the expected value using self.assertAlmostEqual().
        """
        calculated_vanna = vanna(self.S, self.K, self.r, self.y, self.T, self.sigma)
        self.assertAlmostEqual(-0.5270806616216299, calculated_vanna, places=8)

    def test_option_price(self):
        """Tests the calculation of the option price.

        This test function verifies the correctness of the option price calculation by comparing the calculated
        option price to the expected value using self.assertAlmostEqual().
        """
        calculated_option_price = option_price(self.S, self.K, self.r, self.y, self.T, self.sigma, is_call=True)
        self.assertAlmostEqual(11.93852778, calculated_option_price, places=8)

    def test_implied_volatility(self):
        """Tests the calculation of implied volatility.

        This test function validates the correctness of the implied volatility calculation by comparing the
        calculated implied volatility to the expected value using self.assertAlmostEqual().
        """
        calculated_implied_volatility = implied_volatility(10.0, self.S, self.K, self.r, self.y, self.T, is_call=True)[0]
        self.assertAlmostEqual(0.14179029, calculated_implied_volatility, places=8)

    def test_simulate_underlying(self):
        """Tests the simulate_underlying function.

        This test function tests the simulate_underlying function by comparing the simulated asset prices with
        pre-generated data from seeded simulations. It uses np.testing.assert_array_almost_equal() to compare
        the arrays with a specified decimal precision.
        """
        np.random.seed(2023)
        shocks = np.random.normal(size=(2, 2))
        S = simulate_underlying(self.S, self.r, self.y, self.sigma, 0.01, shocks)
        np.testing.assert_array_almost_equal(
            S,
            [[100, 100], [101.44366874, 99.36306716], [99.44117306, 99.84365402]],
            decimal=8
        )
        pip
        install
        inflect
