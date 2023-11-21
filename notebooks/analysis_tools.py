import json
import numpy as np
import matplotlib.pyplot as plt
import stats 

from models import VPC_RNN, VPC_FF

import sys
sys.path.insert(0, "../")
sys.path.insert(0, "../dataset_generator/")

import torch
import trajectory_generator
import environments


def load_parameters(path):
    # load parameters of model
    spec_file = f"{path}/model_parameters.json"
    with open(spec_file, "r") as f:
        params = json.load(f)
    return params

def load_model(path, device = "cpu", model_type = "rnn"):
    # load a model to cpu for detaching + numpy
    params = load_parameters(path)
    model = torch.load(path + f"/trained_{model_type}_model", map_location = torch.device(device))
    model.device = device
    return model, params

def plot_sequence(ratemaps, titles = None, show_units = 5, pane_size = (1, 1), cmap = None, normalize = False):
    """
    Plot sequence ratemaps.
    
    Parameters:
        ratemaps (ndarray): An array of shape (num_envs, num_units, height, width) containing the ratemaps for each environment and unit.
        titles (list, optional): A list of titles for each row of ratemaps. Defaults to None.
        show_units (int, optional): The number of units to show. Defaults to 5.
        pane_size (float, optional): The size of each pane in the plot. Defaults to 1.
        cmap (str, optional): The colormap to use for the plot. Defaults to None.
        normalize (bool, optional): Whether to normalize the ratemaps. Defaults to False.
    
    Returns:
        fig (Figure): The generated matplotlib Figure object.
        axs (ndarray): An array of Axes objects representing the subplots in the plot.
    """
    
    if titles is None:
        titlepad = 0
    else:
        titlepad = 1

    figsize = [pane_size[0]*(len(ratemaps)+titlepad), pane_size[1]*show_units]
    
    fig, axs = plt.subplots(show_units, len(ratemaps) + titlepad, figsize = figsize, squeeze = False)

    max_vals = np.nanmax(ratemaps, axis = (-2, -1))
    
    for i in range(len(ratemaps)):
        # i is column (environment) index!
        for j in range(show_units):
            # j is row (unit) index!
            if normalize:
                vmax = np.amax(max_vals[:,j])
            else:
                vmax = np.amax(max_vals[i,j])

            axs[j,i].imshow(ratemaps[i,j].T, cmap = cmap, origin = "lower", vmin = 0, vmax = vmax)
            axs[j,i].axis("off")
    
    if titles is not None:
        for j in range(show_units):
            axs[j,-1].axis("off")
            axs[j,-1].axis([0, 1, 0, 1])
            axs[j,-1].text(0, 0.4, f"{titles[j]:.1e}")
            
    
    return fig, axs

def plot_ensemble(ratemaps, n = 3, cmap = None, pane_size = (1,1)):
    """
    Plots a grid of ensemble ratemaps.

    Args:
        ratemaps (list): A list of ratemaps to be plotted.
        n (int, optional): The number of rows and columns in the grid. Defaults to 3.
        cmap (str, optional): The colormap to be used for plotting. Defaults to None.
        pane_size (float, optional): The size of each pane in the grid. Defaults to 1.

    Returns:
        fig, ax: The matplotlib figure and axis objects.
    """

    figsize = np.array([n*pane_size[0],n*pane_size[1]])
    fig, ax = plt.subplots(n, n, figsize = figsize)
    
    for i in range(n**2):
        row = i // n
        col = i % n
        ax[row, col].imshow(ratemaps[i].T, origin = "lower", cmap = cmap)
        ax[row, col].axis("off")
    return fig, ax

def spatial_information_selection(p, r, bins, threshold = 0.025):

    info = np.zeros((len(p), p.shape[-1]))

    # compute spatial information
    for i in range(len(p)):
        info[i] = stats.spatial_information(p[i], r[i], bins[i])

    # mask based on cumulative information content
    sorted_inds = np.argsort(info, axis = -1)
    sorted_info = np.array([info[i, sorted_inds[i]] for i in range(len(p))])
    cumulative_info = np.cumsum(sorted_info, axis = -1)/np.sum(sorted_info, axis = -1, keepdims = True)

    # first instance above cutoff
    first = np.argmax(cumulative_info > threshold, axis = -1) 
    cutoffs = [sorted_info[i, first[i]] for i in range(len(p))] # translate into values 

    # create mask
    above_cutoff = [info[i] > cutoffs[i] for i in range(len(p))]
    mask = np.prod(above_cutoff, axis = 0).astype("bool")
    return mask, info

def top_k_sum(x, ax = 0, k = 10):
    """
    find indices of units with top k sum over axis a

    Parameters:
        x (np.ndarray): The input array.
        ax (int, optional): The axis along which the sum is calculated. Default is 0.
        k (int, optional): The number of top sums to return. Default is 10.

    Returns:
        np.ndarray: An array of the top k sums in descending order.
    """
    y = np.sum(x, axis = ax)
    return y.argsort()[-k:][::-1]

def top_k_max(x, ax = 0, k = 10):
    """
    find indices of units with top k max over axis a

    Parameters:
        x (np.ndarray): The input array.
        ax (int, optional): The axis along which the sum is calculated. Default is 0.
        k (int, optional): The number of top sums to return. Default is 10.

    Returns:
        np.ndarray: An array of the top k sums in descending order.
    """
    y = np.amax(x, axis = ax)
    return y.argsort()[-k:][::-1]


def test_dataset(sequence, timesteps, context = False, env = None, device = "cpu", trajectories = True):
    """
    Create a test dataset for a given sequence.

    Parameters:
    - sequence: The sequence of names used to generate the dataset. (type: list[str])
    - timesteps: The number of timesteps in each trajectory. (type: int)
    - context: Whether to include context signals in the input. (default: False) (type: bool)
    - env: The environment object used to generate trajectories. (default: None) (type: object)
    - device: The device to store the tensors on. (default: "cpu") (type: str)
    - trajectories: Whether to generate trajectory paths or individual points. (default: True) (type: bool)

    Returns:
    - x: The model input tensor. (type: tuple[torch.Tensor])
    - rs: np arrays of generated trajectory paths or points. 
    - vs: np arrays of generated velocities. 
    - cs: np arrays of the context signals. 
    """
    # create test dataset
    dg = trajectory_generator.DataGenerator() # generator
    if env is None:
        env = environments.Environments() 
    walls = env.envs

    rs = np.zeros((len(sequence), timesteps, 2), dtype = "float32")
    vs = np.zeros((len(sequence), timesteps, 2), dtype = "float32")
    cs = np.zeros((len(sequence), timesteps, len(walls)), dtype = "float32")

    for i, name in enumerate(sequence):
        walls = env.envs[name]
        
        if trajectories: 
            r, v = dg.generate_paths(1, timesteps + 1, walls)
            vs[i] = v[0]
        else:
            r = dg.generate_points(1, timesteps + 1, walls)
            
        rs[i] = r[0, 1:]
        cs[i] = env.encoding(name)*np.ones((1,timesteps, 1)) # context signal

    if trajectories:
        input_signal = np.copy(vs)
    else:
        input_signal = np.copy(rs)

    if context:
        input_signal = np.concatenate((input_signal, cs), axis = -1)
        
    x = (torch.tensor(input_signal, device = device), torch.tensor(rs, device = device))
    return x, rs, vs, cs # return model input and np versions