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

def plot_sequence(ratemaps, titles = None, show_units = 5, pane_size = 1):
    """Plot sequence experiment.

    Args:
        u (_type_): _description_
        r (_type_): _description_
        bins (_type_): _description_
        titles (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    if titles is None:
        titlepad = 0
    else:
        titlepad = 1
    figsize = [pane_size*(len(ratemaps)+titlepad), pane_size*show_units]
    
    fig, axs = plt.subplots(show_units, len(ratemaps) + titlepad, figsize = figsize, squeeze = False)

    vmax = np.nanmax(ratemaps, axis = (0, -2, -1))
    
    for i in range(len(ratemaps)):
        # i is column (environment) index!
        for j in range(show_units):
            # j is row (unit) index!
            axs[j,i].imshow(ratemaps[i,j].T, cmap = "jet", origin = "lower", vmin = 0)#, vmax = vmax[j])
            axs[j,i].axis("off")
    
    if titles is not None:
        for j in range(show_units):
            axs[j,-1].axis("off")
            axs[j,-1].axis([0, 1, 0, 1])
            axs[j,-1].text(0, 0.4, f"{titles[j]:.1e}")
            
    
    return fig, axs

def plot_ensemble(ratemaps, n = 3, cmap = "jet"):
    """ Plot board of unit activities in a single environment

    Args:
        u (_type_): _description_
        r (_type_): _description_
        bins (_type_): _description_
        n (int, optional): _description_. Defaults to 3.
        cmap (str, optional): _description_. Defaults to "jet".

    Returns:
        _type_: _description_
    """

    figsize = 3*np.array([n,n])
    fig, ax = plt.subplots(n, n, figsize = figsize)
    
    for i in range(n**2):
        row = i // n
        col = i % n
        ax[row, col].imshow(ratemaps[i].T, origin = "lower", cmap = cmap)
        ax[row, col].axis("off")
    return fig, ax

def spatial_information_selection(p, r, bins, threshold = 0.025):
    """_summary_
    Args:
        p (_type_): _description_
        r (_type_): _description_
        bins (_type_): _description_
        threshold (float, optional): _description_. Defaults to 0.025.
    """
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

def test_dataset(sequence, timesteps, context = False, env = None, device = "cpu", trajectories = True):
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