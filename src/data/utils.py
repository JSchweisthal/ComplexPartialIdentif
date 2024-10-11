import torch
from torch.utils.data import Dataset, DataLoader

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data):
        # self.data = df.values
        self.U = data['U']
        self.Z = data['Z']
        self.A = data['A']
        self.X = data['X']
        self.Y = data['Y']
        
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        z = torch.tensor(self.Z[idx], dtype=torch.float32)
        a = torch.tensor(self.A[idx], dtype=torch.long)
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        
        return z, a, x, y

# get datasets
def get_datasets(dataset, seed=None):
    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed) 
    # Create an instance of the dataset

    # create train, validation and test datasets
    train_size = int(0.4 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset

# get dataloaders
    
def get_dataloaders(datasets, batch_size=128, seed=None):
    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed) 
    
    train_dataset, val_dataset, test_dataset = datasets

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader