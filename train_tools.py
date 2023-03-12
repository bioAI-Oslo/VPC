import numpy as np
import torch 
from torch.utils.data import TensorDataset, DataLoader

def l1_reg(l1, vals):
    """L1 regularization
    Args:
        l1 (float): L1 decay parameter
        vals (torch.tensor): Values to be penalized; 
                        can be e.g. weights or activities
    Returns:
        torch float: weighted mean absolute loss
    """
    return l1 * torch.mean(torch.abs(vals))

def l2_reg(l2, vals):
    """L2 regularization
    Args:
        l2 (float): L2 decay parameter
        vals (torch.tensor): Values to be penalized; 
                        can be e.g. weights or activities
    Returns:
        torch float: weighted mean squared loss
    """
    return l2 * torch.mean(vals**2)

def euclid(y, yhat):
    return torch.mean(torch.sqrt(torch.sum((y-yhat)**2, dim = - 1)))

def get_datasets(loc, context, trajectories, batch_size, device = "cpu"):
    
    # load datasets
    train_data = np.load(f"{loc}/train_dataset.npz") 
    val_data = np.load(f"{loc}/val_dataset.npz")    

    if trajectories:
        x_train = train_data["v"]
        x_val = val_data["v"]
    else:
        x_train = train_data["r"]
        x_val = val_data["r"]
        
    y_train = train_data["r"]
    y_val = val_data["r"]

    if context:
        x_train = np.concatenate((x_train, train_data["c"]), axis = -1)
        x_val = np.concatenate((x_val, val_data["c"]), axis = -1)

    train_data = TensorDataset(torch.tensor(x_train, device = device), torch.tensor(y_train, device = device))
    val_data = TensorDataset(torch.tensor(x_val, device = device), torch.tensor(y_val, device = device))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last = True)
    val_loader = DataLoader(val_data, batch_size=batch_size, drop_last = True)
    
    return train_loader, val_loader

class Logger(object):
    """Class for logging and saving metrics
    """
    def __init__(self, dir):
        """
        Args:
            dir (str): path to directory at which to save metrics
        """
        self.dir = dir 
        self.metrics = {}
    
    def __call__(self, new_metrics, partition):
        """Add new metrics to metric history
        Args:
            new_metrics(dict): Metrics at current step, to be saved.
            partition (str): tag to describe metric; typically "val" or "train"
        """
        # log metrics
        for key in new_metrics:
            name = f"{partition}_{key}"
            if not name in self.metrics:
                self.metrics[name] = np.array([]) # add new metric entry if name not found
            self.metrics[name] = np.append(self.metrics[name], new_metrics[key]) # otherwise append to metrics

    def save_metrics(self, name):
        """Unpack and save all metrics
        """
        np.savez(f"{self.dir}/{name}_metrics.npz", **self.metrics)