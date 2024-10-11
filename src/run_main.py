# %%
import os

DATA = 'data1' # 'data1' , 'data2', 'data3'
K_PHI = 2 # 2, 4, 6, 8

SEEDS = [41, 42, 43, 44, 45]
LAMBDA = 1.
GAMMA = 1.
UNOBSERVED_CONFOUNDING = 0.5
NOISE = 0.1


save_folder = 'K_phi'
file_name = str(K_PHI) + '.pt'

save_path = 'results/'
save_path = os.path.join(save_path, DATA + '/', save_folder, file_name)


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import copy
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils import calculate_no_assumption_bounds_multiple_env, generate_one_hot_list

from src.data.utils import CustomDataset, get_datasets, get_dataloaders
from src.models import RepNet, Nuisance1, Nuisance2, Nuisance3

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

if DATA == 'data1':
    from src.data.data1 import generate_data1
    P_Z = 1
elif DATA == 'data2':
    from src.data.data2 import generate_data1
    P_Z = 1
elif DATA == 'data3':
    from src.data.data3 import generate_data1
    P_Z = 20

def get_bounds_naive(x, a, phi, model_pi, model_mu, bounds_support):
    levels_phi = torch.unique(phi)
    mu = torch.tensor([])
    pi = torch.tensor([])
    for p in levels_phi:
        p_val = torch.ones_like(phi) * p
        mu_a = torch.tensor([])
        pi_a = torch.tensor([])
        for av in range(2):
            a_val = torch.ones_like(a) * av
            mu_phi_a = model_mu(x, a_val, p_val)
            pi_phi_a = model_pi(x, p_val)
            pi_phi_a = F.softmax(pi_phi_a, dim=1)
            pi_phi_a = pi_phi_a[:, av].reshape(-1, 1)
            mu_a = torch.cat((mu_a, mu_phi_a), 1)
            pi_a = torch.cat((pi_a, pi_phi_a), 1)
        mu = torch.cat((mu, mu_a.unsqueeze(2)), 2)
        pi = torch.cat((pi, pi_a.unsqueeze(2)), 2)
    mu = torch.permute(mu, (0, 2, 1))
    pi = torch.permute(pi, (0, 2, 1))

    bounds = calculate_no_assumption_bounds_multiple_env(mu, pi, bounds_support, return_tightest=True)

    return bounds

# %%
df_results = pd.DataFrame(columns = ['seed', 'type', 'x', 'z', 'a', 'y', 'upper', 'lower'])

# training loop
for SEED in SEEDS:
    data = generate_data1(2000, UNOBSERVED_CONFOUNDING, NOISE, SEED)
    dataset = CustomDataset(data)
    train_data, val_data, test_data = get_datasets(dataset, seed=SEED)
    datasets = (train_data, val_data, test_data)
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(datasets, batch_size=128, seed=SEED)

    bounds_support = (data['Y'].min(), data['Y'].max())

    z, a, x, y = test_data.dataset[test_data.indices]
    u = torch.from_numpy(test_data.dataset.U[test_data.indices])

    ## Training loop

    # train nuisance models 1
    nuisance1 = Nuisance1(1, P_Z, num_treatments=2, X_dim_hidden=10, Z_dim_hidden=10, shared_dim_hidden=10, 
                        X_layers=2, Z_layers=3, shared_layers=2, lr=0.001)
    early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=10, verbose=False, mode="min")
    trainer = pl.Trainer(max_epochs=100, callbacks=[early_stop_callback], enable_progress_bar=False)
    trainer.fit(nuisance1, train_dataloader, val_dataloader)

    # train nuisance models 2
    nuisance2 = Nuisance2(P_Z, num_treatments=2, Z_dim_hidden=10, 
                        Z_layers=3, lr=0.001)
    early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=10, verbose=False, mode="min")
    trainer = pl.Trainer(max_epochs=100, callbacks=[early_stop_callback], enable_progress_bar=False)
    trainer.fit(nuisance2, train_dataloader, val_dataloader)

    # train nuisance models 3
    nuisance3 = Nuisance3(1, P_Z, num_treatments=2, X_dim_hidden=10, Z_dim_hidden=10, shared_dim_hidden=10, outcome_dim=10, 
                        X_layers=2, Z_layers=3, shared_layers=2, outcome_layers=2, lr=0.001)
    early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=10, verbose=False, mode="min")
    trainer = pl.Trainer(max_epochs=100, callbacks=[early_stop_callback], enable_progress_bar=False)
    trainer.fit(nuisance3, train_dataloader, val_dataloader)

    # RepNet
    repnet = RepNet(1, P_Z, nuisance1, nuisance2, nuisance3, num_treatments=2,
                    num_environments=K_PHI, Z_dim_hidden=10, Z_layers=3, 
                    lr=0.03,
                    gamma=GAMMA, alpha=LAMBDA,
                    bounds_support=bounds_support)
    early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=50, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val/loss", mode="min")
    trainer = pl.Trainer(max_epochs=100, enable_progress_bar=False, 
                        callbacks=[early_stop_callback, checkpoint_callback]) 
    trainer.fit(repnet, train_dataloader, val_dataloader)
    # repnet best model

    repnet = RepNet.load_from_checkpoint(checkpoint_callback.best_model_path, X_dim=1, Z_dim=P_Z, nuisance1=nuisance1, nuisance2=nuisance2,
                                    nuisance3= nuisance3, num_treatments=2,
                    num_environments=K_PHI, Z_dim_hidden=10, Z_layers=3, 
                    lr=0.03,
                    gamma=GAMMA, alpha=LAMBDA,
                    bounds_support=bounds_support)

    # eval
    mu, pi, phi, cls = repnet(x, z, a)
    bounds = repnet.get_bounds(x, z, a, return_tightest=True)

    ############ 
    # baseline

    kmeans = KMeans(n_clusters=K_PHI)

    kmeans = kmeans.fit(data['Z']) 

    phi_naive_all = kmeans.labels_
    phi_norm = (phi_naive_all - phi_naive_all.min())/(phi_naive_all.max() - phi_naive_all.min()) - 0.5
    data_aux_naive = copy.deepcopy(data)
    data_aux_naive['Z'] = torch.from_numpy(phi_norm).reshape(-1, 1)
    dataset_naive = CustomDataset(data_aux_naive)
    train_data_naive, val_data_naive, test_data_naive = get_datasets(dataset_naive, seed=SEED)
    datasets_naive = (train_data_naive, val_data_naive, test_data_naive)
    train_dataloader_naive, val_dataloader_naive, test_dataloader_naive = get_dataloaders(datasets_naive, batch_size=128, seed=SEED)

    # train nuisance models 2
    model_pi_naive = Nuisance1(1, 1, num_treatments=2, X_dim_hidden=10, Z_dim_hidden=10, shared_dim_hidden=10, 
                        X_layers=2, Z_layers=3, shared_layers=2, lr=0.001)
    early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=10, verbose=False, mode="min")
    trainer = pl.Trainer(max_epochs=100, callbacks=[early_stop_callback], enable_progress_bar=False)
    trainer.fit(model_pi_naive, train_dataloader_naive, val_dataloader_naive)

    model_mu_naive = Nuisance3(1, 1, num_treatments=2, X_dim_hidden=10, Z_dim_hidden=10, shared_dim_hidden=10, outcome_dim=10,
                            X_layers=2, Z_layers=3, shared_layers=2, outcome_layers=2, lr=0.001)
    early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=10, verbose=False, mode="min")
    trainer = pl.Trainer(max_epochs=100, callbacks=[early_stop_callback], enable_progress_bar=False)
    trainer.fit(model_mu_naive, train_dataloader_naive, val_dataloader_naive)

    phi_naive = torch.from_numpy(phi_naive_all[test_data.indices])

    bounds_naive = get_bounds_naive(x, a, phi_naive.reshape(-1, 1).float(), model_pi_naive, model_mu_naive, bounds_support)

    df_bounds_plot = pd.DataFrame(columns = ['seed', 'type', 'x', 'z', 'a', 'y', 'upper', 'lower'])
    df_bounds_plot['x'] = x.reshape(-1)
    df_bounds_plot['z'] = z.sum(1).reshape(-1)
    df_bounds_plot['a'] = a.reshape(-1)
    df_bounds_plot['y'] = y.reshape(-1)
    df_bounds_plot['upper'] = bounds[:, 1].detach().numpy().reshape(-1)
    df_bounds_plot['lower'] = bounds[:, 0].detach().numpy().reshape(-1)
    df_bounds_plot['type'] = 'ours'

    df_bounds_plot_naive = pd.DataFrame(columns = ['seed', 'type', 'x', 'z', 'a', 'y', 'upper', 'lower'])
    df_bounds_plot_naive['x'] = x.reshape(-1)
    df_bounds_plot_naive['z'] = z.sum(1).reshape(-1)
    df_bounds_plot_naive['a'] = a.reshape(-1)
    df_bounds_plot_naive['y'] = y.reshape(-1)
    df_bounds_plot_naive['upper'] = bounds_naive[:, 1].detach().numpy().reshape(-1)
    df_bounds_plot_naive['lower'] = bounds_naive[:, 0].detach().numpy().reshape(-1)
    df_bounds_plot_naive['type'] = 'naive'

    df_bounds_plot = pd.concat([df_bounds_plot, df_bounds_plot_naive])

    df_bounds_plot['seed'] = SEED

    df_results = df_results.append(df_bounds_plot, ignore_index=True)

    print(SEED)

num_x = 100
df_results['x_plt'] = (df_results['x'] * num_x).round() / num_x


dict_save = {'df_results': df_results.to_dict(), 
             'SEEDS': SEEDS, 
             'K_PHI': K_PHI, 
             'LAMBDA': LAMBDA, 
             'UNOBSERVED_CONFOUNDING': UNOBSERVED_CONFOUNDING, 
             'NOISE': NOISE}

torch.save(dict_save, save_path)

print('Results saved in: ', save_path)
