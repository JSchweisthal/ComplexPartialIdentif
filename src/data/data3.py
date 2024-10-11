import numpy as np
import pandas as pd
P_Z = 20

def propensity_score(X, U, Z, strength_confounding):    
    Z[:, int(Z.shape[1]/4):] = 0
    coeff_Z = (Z.sum(1) - Z.shape[1] * 0.5) / (Z.shape[1] * 0.5 * ( 1 - 0.5))
    ps = (np.sin(10*coeff_Z + X.mean(1) + U.mean(1))*0.48 + 0.48  + 0.04/(1+np.exp(-3*np.abs(5*coeff_Z)))) 
    return ps

# Generate CATE
def cate(X):
    return ( -(1.6* X.mean(1)+0.5)**4 + 12 *np.sin(4*X.mean(1)+1.5) +  1 * np.cos(1*X.mean(1)))/10 /8 *(-1) +0.5

# Generate outcome Y
def outcome(X, A, U, strength_confounding, noise):
    Y = X.mean(1) + strength_confounding * U.mean(1) + noise * np.random.laplace(0, 1, size=X.shape[0])
    Y = Y * (0.25) + cate(X) * A.reshape(-1) 
    return Y.reshape(-1, 1)

def delta_Z(X, p_z=P_Z):
    Z = np.random.binomial(1, 0.5, size=(X.shape[0], p_z)).astype(float)
    return Z

def generate_x(num_samples):
    return np.random.uniform(-1, 1, size=(num_samples, 1))

def generate_u(num_samples):
    return np.random.normal(-1, 1, size=(num_samples, 1))

def generate_data1(num_samples, strength_confounding, noise, seed):
    # Set random seed for reproducibility
    np.random.seed(seed)
    # Generate observed confounders X
    X = generate_x(num_samples)
    X = generate_x(num_samples)
    del_Z = delta_Z(X)
    p_z = del_Z.shape[1]
    Z = del_Z

    # Generate unobserved confounder U
    U = generate_u(num_samples)

    ps = propensity_score(X, U, Z, strength_confounding)
    A = np.random.binomial(1, ps, size=num_samples).reshape(-1, 1)

    Y = outcome(X, A, U, strength_confounding, noise)

    # data
    data = {'X': X, 'U': U, 'A': A, 'Y': Y, 'Z': Z}
    return data


    
