import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_no_assumption_bounds(potential_outcomes, propensity_scores, trivial_bounds):
    """
    Calculate the "no assumption bounds" for treatment effect estimation.
    
    Args:
    - potential_outcomes (numpy array): Array of potential outcomes for each treatment and environment.
    - propensity_scores (numpy array): Array of propensity scores for each treatment and environment.
    - trivial_bounds (tuple): Tuple of trivial bounds for the treatment effect.
    
    Returns:
    - no_assumption_bounds (tuple): Tuple of lower and upper bounds for the treatment effect.
    """
    # Calculate the upper bound
    upper_bound = propensity_scores[:, 1] * potential_outcomes[:, 1] + (1-propensity_scores[:, 1]) * trivial_bounds[1]\
    - (1-propensity_scores[:, 0]) * trivial_bounds[0] - propensity_scores[:, 0] * potential_outcomes[:, 0]

    # Calculate the lower bound
    lower_bound = propensity_scores[:, 1] * potential_outcomes[:, 1] + (1-propensity_scores[:, 1]) * trivial_bounds[0]\
    - (1-propensity_scores[:, 0]) * trivial_bounds[1] - propensity_scores[:, 0] * potential_outcomes[:, 0]
    
    # Return the no assumption bounds
    no_assumption_bounds = torch.stack((lower_bound, upper_bound), dim=1)
    return no_assumption_bounds


def calculate_no_assumption_bounds_multiple_env(potential_outcomes, propensity_scores, trivial_bounds, idx_treatments=[0, 1], return_tightest=False):
    """
    Calculate the "no assumption bounds" for treatment effect estimation for multiple environments.
    
    Args:
    - potential_outcomes (torch.Tensor): Tensor of potential outcomes for each treatment and environment.
    - propensity_scores (torch.Tensor): Tensor of propensity scores for each treatment and environment.
    - trivial_bounds (tuple): Tuple of trivial bounds for the treatment effect.
    
    Returns:
    - no_assumption_bounds (torch.Tensor): Tensor of lower and upper bounds for the treatment effect.
    """
    num_environments = potential_outcomes.shape[1]
    num_treatments = potential_outcomes.shape[2]
    
    no_assumption_bounds = torch.tensor([])
    # Calculate the lower and upper bounds for each combinations of environments
    for e1 in range(num_environments):
        for e2 in range(num_environments):
            potential_outcomes_et = potential_outcomes[:, [e1, e2], idx_treatments]
            propensity_scores_et = propensity_scores[:, [e1, e2], idx_treatments]
            no_assumption_bounds_et = calculate_no_assumption_bounds(potential_outcomes_et, propensity_scores_et, trivial_bounds)
            no_assumption_bounds = torch.cat((no_assumption_bounds, no_assumption_bounds_et.unsqueeze(dim=1)), dim=1)
    
    if return_tightest:
        # Choose the tightest bounds
    # Choose the tightest bounds
        lower_bound = no_assumption_bounds[:, :, 0].max(dim=1)[0]
        upper_bound = no_assumption_bounds[:, :, 1].min(dim=1)[0]

        
        # Return the no assumption bounds
        no_assumption_bounds = torch.stack((lower_bound, upper_bound), dim=1)
        return no_assumption_bounds
    else:
        bounds_wb = (no_assumption_bounds[:, 0, :], no_assumption_bounds[:, 3, :])
        bounds_cb = (no_assumption_bounds[:, 2, :], no_assumption_bounds[:, 1, :])
        return bounds_wb, bounds_cb

def generate_one_hot_list(num_environments):
    # Create an empty list to hold the one-hot encoded tensors
    one_hot_list = []

    # Loop through each environment index and create a one-hot encoded tensor
    for i in range(num_environments):
        # Create a tensor where all elements are zero except the i-th element
        one_hot_tensor = torch.zeros(num_environments, dtype=torch.float32)
        one_hot_tensor[i] = 1.
        # Append the one-hot encoded tensor to the list
        one_hot_list.append(one_hot_tensor)

    return one_hot_list