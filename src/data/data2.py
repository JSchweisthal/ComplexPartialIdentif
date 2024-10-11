import numpy as np
import pandas as pd

def propensity_score(X, U, Z, strength_confounding):
    ps = (np.sin(2.5*Z.mean(1) + X.mean(1) + U.mean(1))*0.48 + 0.48  + 0.04/(1+np.exp(-3*np.abs(Z.mean(1))))) 
    return ps

# Generate CATE
def cate(X):
    return ( (2.5* X.mean(1))**4 + 12 *np.sin(6*X.mean(1)) +  0.5 * np.cos(1*X.mean(1)))/10 /8 *(-1) +0.5

# Generate outcome Y
def outcome(X, A, U, strength_confounding, noise):
    Y = X.mean(1) + strength_confounding * U.mean(1) + noise * np.random.laplace(0, 1, size=X.shape[0])
    Y = Y * (0.25) + cate(X) * A.reshape(-1) 
    return Y.reshape(-1, 1)

def delta_Z(X, noise=0.1):
    # samples out of mixture distribution: 1/3 prob from uniform -1, 1, 1/3 from beta(2, 2), 1/3 from beta(2, 2) *-1
    Z_1 = np.random.uniform(-1, 1, size=(X.shape[0], 1))
    Z_2 = np.random.beta(2, 2, size=(X.shape[0], 1))
    Z_3 = -1 * Z_2
    idx_Z = np.random.choice([0, 1, 2], p=[0.5, 0.25, 0.25], size=X.shape[0])
    Z = np.array([Z_1[i] if idx_Z[i] == 0 else Z_2[i] if idx_Z[i] == 1 else Z_3[i] for i in range(X.shape[0])])
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
    Z = del_Z

    # Generate unobserved confounder U
    U = generate_u(num_samples)

    ps = propensity_score(X, U, Z, strength_confounding)
    A = np.random.binomial(1, ps, size=num_samples).reshape(-1, 1)

    Y = outcome(X, A, U, strength_confounding, noise)

    # data
    data = {'X': X, 'U': U, 'A': A, 'Y': Y, 'Z': Z}
    return data
    
