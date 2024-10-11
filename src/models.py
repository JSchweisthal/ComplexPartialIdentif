import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import copy
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils import calculate_no_assumption_bounds_multiple_env, generate_one_hot_list

# representation network
# takes as input the 3 objects of the 3 nuisance models from below
# predicts the optimal bounds and learns discrete representations of the instrument Z which minimze the bounds width

class RepNet(pl.LightningModule):
    def __init__(self, X_dim, Z_dim,  nuisance1, nuisance2, nuisance3,
                  num_treatments=2, num_environments=2,
                  Z_dim_hidden=10, Z_layers=1, bounds_support=(0, 1),
                   gamma=1., alpha=1.,  lr=0.001):
        super(RepNet, self).__init__()
        self.X_dim = X_dim
        self.Z_dim = Z_dim
        self.nuisance1 = nuisance1
        self.nuisance2 = nuisance2
        self.nuisance3 = nuisance3
        self.num_treatments = num_treatments
        self.num_environments = num_environments
        self.Z_dim_hidden = Z_dim_hidden
        self.Z_layers = Z_layers
        self.lr = lr

        self.bounds_support = bounds_support
        self.gamma = gamma
        self.alpha = alpha

        # self.size = 1

        nuisance1.eval()
        nuisance2.eval()
        nuisance3.eval()

        # representation network
        instrument_layers = nn.ModuleList([nn.Linear(Z_dim, Z_dim_hidden)])
        for i in range(self.Z_layers-1):
            instrument_layers.append(nn.Linear(Z_dim_hidden, Z_dim_hidden))
            instrument_layers.append(nn.ReLU())
        self.instrument_layers = nn.Sequential(*instrument_layers)
        self.discretization_layer = nn.Linear(Z_dim_hidden, num_environments)

        # classification layer
        self.classification_layer = nn.Linear(Z_dim_hidden, num_environments)

    def get_representation(self, z, hard=True):
        psi = self.instrument_layers(z)
        phi = self.discretization_layer(psi)
        phi = F.gumbel_softmax(phi, tau=1, hard=hard)
        cls = self.classification_layer(psi)
        return phi, cls
    
    def get_mu_pi(self, x, z_pop, a_pop, phi, a, phi_j=None):
        epsilon = 0.0001
        if phi_j is None:
            phi_j, cls = self.get_representation(z_pop, hard=True)
        n = x.shape[0]
        indices = torch.cartesian_prod(torch.arange(n), torch.arange(n))
        ###
        z_ext = z_pop[indices[:, 1]].reshape(n**2, z_pop.shape[1])
        phi_j_ext = phi_j[indices[:, 1]].reshape(n**2, phi_j.shape[1])
        x_ext = x[indices[:, 0]]
        a_pop_ext = a_pop[indices[:, 1]].reshape(n**2, a_pop.shape[1])
        phi = phi[indices[:, 0]]
        a_ext = a[indices[:, 0]]
        ###
        index_a = a_ext.reshape(-1)
        index_a = F.one_hot(index_a, num_classes=2)
        A_xz = self.nuisance1(x_ext, z_ext).detach()
        A_xz = F.softmax(A_xz, dim=1)
        A_xz = A_xz[index_a.bool()].reshape(-1, 1)
        A_z = self.nuisance2(z_ext).detach()
        A_z = F.softmax(A_z, dim=1)
        A_z = A_z[index_a.bool()].reshape(-1, 1)
        Y_xaz = self.nuisance3(x_ext, a_ext, z_ext).detach()
        a_pop_ext = a_pop_ext.float()
        averages = phi_j_ext.sum(0, keepdim=True).t() / n**2

        prob_phi = phi_j_ext.matmul(averages)  
        sums_a = phi_j_ext.t().matmul(a_pop_ext)  
        counts_phi = phi_j_ext.sum(0, keepdim=True).t()  
        averages_a = sums_a / torch.clamp(counts_phi, min=1)
        prob_a_phi = phi_j_ext.matmul(averages_a)

        prob_phi = torch.clamp(prob_phi, min=epsilon, max=1-epsilon)
        prob_a_phi = torch.clamp(prob_a_phi, min=epsilon, max=1-epsilon)
        
        pi = A_xz * (1-(phi_j_ext-phi).max(1)[0].reshape(-1, 1))/prob_phi
        mu = Y_xaz * ((1-(phi_j_ext-phi).max(1)[0].reshape(-1, 1))*A_z) /(prob_a_phi * prob_phi)

        pi = pi.reshape(n, n).mean(1).reshape(n, 1)
        mu = mu.reshape(n, n).mean(1).reshape(n, 1)

        return mu, pi
    
    def forward(self, x, z_pop, a_pop):
        phi_j, cls = self.get_representation(z_pop, hard=True)
        phi_vals = generate_one_hot_list(self.num_environments)
        mu = torch.tensor([])
        pi = torch.tensor([])
        for phi_opt in phi_vals: 
            phi = phi_opt.repeat(x.shape[0], 1)
            mu_phi = torch.tensor([])
            pi_phi = torch.tensor([])
            for a_opt in [torch.tensor([0]), torch.tensor([1])]:
                a = a_opt.repeat(x.shape[0], 1)
                mu_phi_a, pi_phi_a = self.get_mu_pi(x, z_pop, a_pop, phi, a, phi_j)
                mu_phi = torch.cat((mu_phi, mu_phi_a), 1)
                pi_phi = torch.cat((pi_phi, pi_phi_a), 1)
            mu = torch.cat((mu, mu_phi.unsqueeze(2)), 2)
            pi = torch.cat((pi, pi_phi.unsqueeze(2)), 2)

        mu = torch.permute(mu, (0, 2, 1))
        pi = torch.permute(pi, (0, 2, 1))

        
        return mu, pi, phi_j, cls
    
    def get_bounds(self, x, z_pop, a_pop, bounds_support=None, return_tightest=True):
        if bounds_support is None:
            bounds_support = self.bounds_support
        mu, pi, phi, cls = self.forward(x, z_pop, a_pop)
        epsilon = 0.0001
        mu = torch.clamp(mu, min=bounds_support[0], max=bounds_support[1])
        pi = torch.clamp(pi, min=epsilon, max=1-epsilon)
        
        bounds = calculate_no_assumption_bounds_multiple_env(mu, pi, bounds_support, return_tightest=return_tightest)
        return bounds
    
    def loss(self, mu, pi, bounds_support, gamma):
        epsilon = 0.0001
        mu = torch.clamp(mu, min=bounds_support[0], max=bounds_support[1])
        pi = torch.clamp(pi, min=epsilon, max=1-epsilon)
        
        bounds = calculate_no_assumption_bounds_multiple_env(mu, pi, bounds_support, return_tightest=True)
        lower_bound = bounds[:, 0]
        upper_bound = bounds[:, 1]
        loss = (upper_bound - lower_bound).mean() / (bounds_support[1]-bounds_support[0]) # + gamma * marginal_entropy
        return loss
    
    def training_step(self, batch, batch_idx):
        Z, A, X, Y = batch
        mu, pi, phi, cls = self.forward(X, Z, A)
        loss = self.loss(mu, pi, self.bounds_support, self.gamma)
        loss_cls = F.cross_entropy(cls, phi)
        eps = 0.0001
        loss_bal = - torch.log(phi.mean(0)+eps).mean() #((1-phi.mean(0)) / phi.mean(0)+ eps).mean()
        loss = loss + self.gamma * loss_cls + self.alpha * loss_bal
        if torch.isnan(loss).any():
            pass
        else:
            self.log('train/loss', loss, on_epoch=True, on_step=False)
            return loss
    
    def validation_step(self, batch, batch_idx):
        Z, A, X, Y = batch
        mu, pi, phi, cls = self.forward(X, Z, A)
        loss = self.loss(mu, pi, self.bounds_support, self.gamma)
        ###################################
        loss_cls = F.cross_entropy(cls, phi)
        eps = 0.0001
        loss_bal = - torch.log(phi.mean(0)+eps).mean() 
        loss = loss + self.gamma * loss_cls + self.alpha * loss_bal
        ###################################
        if torch.isnan(loss).any():
            self.log('val/loss', loss, on_epoch=True, on_step=False)
        #     pass
        else:
            self.log('val/loss', loss, on_epoch=True, on_step=False)

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    

# nuisance model1:
# a simple nuisance model that classifies the treatment A given confounders X and instrument Z
class Nuisance1(pl.LightningModule):
    def __init__(self, X_dim, Z_dim, num_treatments=2,
                 X_dim_hidden=10, Z_dim_hidden=10, shared_dim_hidden=10,
                 X_layers=1, Z_layers=1, shared_layers=1, lr=0.001):
        super(Nuisance1, self).__init__()
        self.X_dim = X_dim
        self.Z_dim = Z_dim
        self.num_treatments = num_treatments
        self.X_dim_hidden = X_dim_hidden
        self.Z_dim_hidden = Z_dim_hidden
        self.shared_dim_hidden = shared_dim_hidden
        self.X_layers = X_layers
        self.Z_layers = Z_layers
        self.shared_layers = shared_layers
        self.lr = lr

        # X representation layers
        X_layers = nn.ModuleList([nn.Linear(X_dim, X_dim_hidden)])
        for i in range(self.X_layers-1):
            X_layers.append(nn.Linear(X_dim_hidden, X_dim_hidden))
            X_layers.append(nn.ReLU())
        X_layers.append(nn.Linear(X_dim_hidden, X_dim_hidden))
        self.X_layers = nn.Sequential(*X_layers)

        # Z representation layers
        Z_layers = nn.ModuleList([nn.Linear(Z_dim, Z_dim_hidden)])
        for i in range(self.Z_layers-1):
            Z_layers.append(nn.Linear(Z_dim_hidden, Z_dim_hidden))
            Z_layers.append(nn.ReLU())
        Z_layers.append(nn.Linear(Z_dim_hidden, Z_dim_hidden))
        self.Z_layers = nn.Sequential(*Z_layers)

        # shared representation layers
        shared_layers = nn.ModuleList([nn.Linear(X_dim_hidden + Z_dim_hidden, shared_dim_hidden)])
        for i in range(self.shared_layers-1):
            shared_layers.append(nn.Linear(shared_dim_hidden, shared_dim_hidden))
            shared_layers.append(nn.ReLU())
        shared_layers.append(nn.Linear(shared_dim_hidden, num_treatments))
        self.shared_layers = nn.Sequential(*shared_layers)

        self.save_hyperparameters()

    def forward(self, X, Z):
        X = self.X_layers(X)
        Z = self.Z_layers(Z)
        shared = torch.cat((X, Z), dim=1)
        A_hat = self.shared_layers(shared)
        return A_hat
    
    def loss(self, A_hat, A):
        A = A.reshape(-1)
        return F.cross_entropy(A_hat, A)
    
    def training_step(self, batch, batch_idx):
        Z, A, X, Y = batch
        A_hat = self(X, Z)
        loss = self.loss(A_hat, A)
        self.log('train/loss', loss, on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        Z, A, X, Y = batch
        A_hat = self(X, Z)
        loss = self.loss(A_hat, A)
        self.log('val/loss', loss, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
# nuisance model2:
# a simple nuisance model that classifies the treatment A given instrument Z

class Nuisance2(pl.LightningModule):
    def __init__(self, Z_dim, num_treatments=2,
                 Z_dim_hidden=10,
                 Z_layers=1, lr=0.001):
        super(Nuisance2, self).__init__()
        self.Z_dim = Z_dim
        self.num_treatments = num_treatments
        self.Z_dim_hidden = Z_dim_hidden
        self.Z_layers = Z_layers
        self.lr = lr

        # Z representation layers
        Z_layers = nn.ModuleList([nn.Linear(Z_dim, Z_dim_hidden)])
        for i in range(self.Z_layers-1):
            Z_layers.append(nn.Linear(Z_dim_hidden, Z_dim_hidden))
            Z_layers.append(nn.ReLU())
        Z_layers.append(nn.Linear(Z_dim_hidden, num_treatments))
        self.Z_layers = nn.Sequential(*Z_layers)

        self.save_hyperparameters()


    def forward(self, Z):
        A_hat = self.Z_layers(Z)
        return A_hat
    
    def loss(self, A_hat, A):
        A = A.reshape(-1)
        return F.cross_entropy(A_hat, A)
    
    def training_step(self, batch, batch_idx):
        Z, A, X, Y = batch
        A_hat = self(Z)
        loss = self.loss(A_hat, A)
        self.log('train/loss', loss, on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        Z, A, X, Y = batch
        A_hat = self(Z)
        loss = self.loss(A_hat, A)
        self.log('val/loss', loss, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

# nuisance model3:
# a simple nuisance model that predicts the outcome Y given confounders X, treatment A, and instrument Z
# using shared layers after representations of X and Z and then own outcome layers per treatment A

class Nuisance3(pl.LightningModule):
    def __init__(self, X_dim, Z_dim, num_treatments=2,
                 X_dim_hidden=10, Z_dim_hidden=10, shared_dim_hidden=10, outcome_dim=10,
                 X_layers=1, Z_layers=1, shared_layers=1, outcome_layers=1, lr=0.001):
        super(Nuisance3, self).__init__()
        self.X_dim = X_dim
        self.Z_dim = Z_dim
        self.num_treatments = num_treatments
        self.X_dim_hidden = X_dim_hidden
        self.Z_dim_hidden = Z_dim_hidden
        self.shared_dim_hidden = shared_dim_hidden
        self.outcome_dim = outcome_dim
        self.X_layers = X_layers
        self.Z_layers = Z_layers
        self.shared_layers = shared_layers
        self.outcome_layers = outcome_layers
        self.lr = lr

        # X representation layers
        X_layers = nn.ModuleList([nn.Linear(X_dim, X_dim_hidden)])
        for i in range(self.X_layers-1):
            X_layers.append(nn.Linear(X_dim_hidden, X_dim_hidden))
            X_layers.append(nn.ReLU())
        X_layers.append(nn.Linear(X_dim_hidden, X_dim_hidden))
        self.X_layers = nn.Sequential(*X_layers)

        # Z representation layers
        Z_layers = nn.ModuleList([nn.Linear(Z_dim, Z_dim_hidden)])
        for i in range(self.Z_layers-1):
            Z_layers.append(nn.Linear(Z_dim_hidden, Z_dim_hidden))
            Z_layers.append(nn.ReLU())
        Z_layers.append(nn.Linear(Z_dim_hidden, Z_dim_hidden))
        self.Z_layers = nn.Sequential(*Z_layers)

        # shared representation layers
        shared_layers = nn.ModuleList([nn.Linear(X_dim_hidden + Z_dim_hidden, shared_dim_hidden)])
        for i in range(self.shared_layers-1):
            shared_layers.append(nn.Linear(shared_dim_hidden, shared_dim_hidden))
            shared_layers.append(nn.ReLU())
        shared_layers.append(nn.Linear(shared_dim_hidden, shared_dim_hidden))
        self.shared_layers = nn.Sequential(*shared_layers)

        # outcome heads per treatment
        outcome_layers = nn.ModuleList([])
        for t in range(num_treatments):
            outcome_layers_t = nn.ModuleList([nn.Linear(shared_dim_hidden, outcome_dim), nn.ReLU()])
            for i in range(self.outcome_layers-1):
                outcome_layers_t.append(nn.Linear(outcome_dim, outcome_dim))
                outcome_layers_t.append(nn.ReLU())
            outcome_layers_t.append(nn.Linear(outcome_dim, 1))
            outcome_layers_t = (nn.Sequential(*outcome_layers_t))
            outcome_layers.append(outcome_layers_t)
        self.outcome_layers = outcome_layers

        self.save_hyperparameters()

    def get_potential_outcomes(self, X, Z):
        X = self.X_layers(X)
        Z = self.Z_layers(Z)
        shared = torch.cat((X, Z), dim=1)
        shared = self.shared_layers(shared)
        outcomes = []
        for t in range(self.num_treatments):
            outcome = self.outcome_layers[t](shared)
            outcomes.append(outcome)
        outcomes = torch.stack(outcomes)
        outcomes = torch.permute(outcomes, (1, 0, 2)).squeeze()
        return outcomes

    def forward(self, X, A, Z):
        outcomes = self.get_potential_outcomes(X, Z)
        A = A.reshape(-1)
        factual_outcomes = outcomes[torch.arange(outcomes.size(0)), A]
        factual_outcomes = factual_outcomes.reshape(-1, 1)
        return factual_outcomes
    
    def loss(self, outcomes, Y):
        # only select factual outcomes
        loss = F.mse_loss(outcomes, Y)
        return loss
    
    def training_step(self, batch, batch_idx):
        Z, A, X, Y = batch
        outcomes = self(X, A, Z)
        loss = self.loss(outcomes, Y)
        self.log('train/loss', loss, on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        Z, A, X, Y = batch
        outcomes = self(X, A, Z)
        loss = self.loss(outcomes, Y)
        self.log('val/loss', loss, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    

