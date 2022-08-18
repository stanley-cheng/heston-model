''' 
Module to calculate the Heston price of a European call or put option. It does so by calculating the integrand and numerically integrating by using Gauss-Legendre Quadrature.
'''
import numpy as np

i = np.complex(0, 1)

def integrand(phi, S, K, tau, r, q, v0, kappa, theta, sigma, rho):
    '''
    Calculates the real part of the integrand in the equation for the Heston price.

    Parameters
    ----------
    phi: float
        Dummy variable for integration
    S: float
        Spot price of the underlying asset
    K: float
        Strike price of the option
    tau: float
        Time to maturity in years
    r: float
        Risk free interest rate
    q: float
        Dividend rate
    v0: float
        Initial variance
    kappa: float
        Rate of mean reversion (rate of which the variance at time t reverts to theta)
    theta: float
        Long variance, or the long run average of the price
    sigma: float
        Volatility of the volatility
    rho:
        Correlation coefficient of the two Wiener processes (the spot price and the volatility)

    Returns
    ----------
    integrand: float
        Value of the integrand which will be used for numerical integration

    '''

    x = np.log(S)

    # First characteristic function
    u1 = 0.5
    b1 = kappa + lambd - rho * sigma
    d1 = np.sqrt((rho * sigma * i * phi - b1) * (rho * sigma * i * phi - b1) - sigma * sigma * (2 * u1 * i * phi - phi * phi))
    c1 = (b1 - rho * sigma * i * phi - d1) / (b1 - rho * sigma * i * phi + d1)
    C1 = (r-q) * i * phi * tau + (kappa * theta) / (sigma * sigma) * ((b1 - rho * sigma * i * phi - d1) * tau - 2 * np.log((1 - c1 * np.exp(-d1 * tau)) / (1 - c1)))
    D1 = ((b1 - rho * sigma * i * phi - d1) / (sigma * sigma)) * ((1 - np.exp(-d1 * tau)) / (1 - c1 * np.exp(-d1 * tau)))
    f1 = np.exp(C1 + D1 * v0 + i * phi * x)

    # Second characteristic function
    u2 = -0.5
    b2 = kappa + lambd
    d2 = np.sqrt((rho * sigma * i * phi - b2) * (rho * sigma * i * phi - b2) - sigma * sigma * (2 * u2 * i * phi - phi * phi))
    c2 = (b2 - rho * sigma * i * phi - d2) / (b2 - rho * sigma * i * phi + d2)
    C2 = (r-q) * i * phi * tau + (kappa * theta) / (sigma * sigma) * ((b2 - rho * sigma * i * phi - d2) * tau - 2 * np.log((1 - c2 * np.exp(-d2 * tau)) / (1 - c2)))
    D2 = ((b2 - rho * sigma * i * phi - d2) / (sigma * sigma)) * ((1 - np.exp(-d2 * tau)) / (1 - c2 * np.exp(-d2 * tau)))
    f2 = np.exp(C2 + D2 * v0 + i * phi * x)
    
    part1 = S * np.exp(-q * tau) * f1
    part2 = K * np.exp(-r * tau) * f2


    return np.real((np.exp(-i * phi * np.log(K)) / (i * phi)) * (part1 - part2))


def heston_price(CP, S, K, tau, r, q, v0, kappa, theta, sigma, rho):
    '''
    Calculates the Heston price of a European option.

    Parameters
    ----------
    CP: str
        'c' or 'p' for call or put option respectively
    S: float
        Spot price of the underlying asset
    K: float
        Strike price of the option
    tau: float
        Time to maturity in years
    r: float
        Risk free interest rate
    q: float
        Dividend rate
    v0: float
        Initial variance
    kappa: float
        Rate of mean reversion (rate of which the variance at time t reverts to theta)
    theta: float
        Long variance, or the long run average of the price
    sigma: float
        Volatility of the volatility
    rho:
        Correlation coefficient of the two Wiener processes (the spot price and the volatility)

    Returns
    ----------
    price: float
        Heston price of the option

    '''

    main1 = S * np.exp(-q * tau) - K * np.exp(-r * tau)
    
    # Abscissas and weights using Gauss-Legendre Quadrature
    x, w = np.polynomial.legendre.leggauss(32) # this number is the order of the polynomial used, the higher the more accurate the result but the longer the computational time

    # Setting bounds of integration
    a, b = 0, 100 # the upper bound should be set as close to infinity as possible, but as the integrand function is rapidly decaying, 100 is sufficient
    
    # Carrying out integration by Gauss-Legendre Quadrature
    phi = ((b-a) * 0.5 * x) + (a + b) * 0.5
    integral = np.sum(w * integrand(phi, S, K, tau, r, q, v0, kappa, theta, sigma, rho))
    trans_integral = (b - a) * 0.5 * integral
    
    C = 0.5 * main1 + trans_integral / np.pi

    if CP == 'c':
        return C
    elif CP == 'p':
        return C - main1
    else:
        return TypeError('CP must have values "c" or "p"')


