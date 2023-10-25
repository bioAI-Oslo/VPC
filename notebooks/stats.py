import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.stats
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve

def ratemaps(s, r, bins, smooth = False):
    """ Vector of ratemaps

    Args:
        s (np.array): signal to be binned. Of shape (N, T, Nc)
        r (np.array): coordinates corresponding to signal values. Of shape (N, T, 2)
            N is the number of samples, Nc the number of units in the population.
        bins: Bins used for ratemap construction. See binned_statistic_2d docs.
        smooth (bool, optional): Whether to smooth ratemaps using a Gaussian kernel with NaN interpolation. 
            Defaults to False.
    Returns:
        np.array: Population vector of ratemaps, of shape (Nc, binx, biny)
    """
    x = r[...,0] 
    y = r[...,1]
    
    ratemaps = []
    
    for i in range(len(s)): 
        stack, _,_,_ = scipy.stats.binned_statistic_2d(x[i], y[i], s[i].T, bins = bins[i]) 
        if smooth:
            kernel = Gaussian2DKernel(x_stddev=1)._array # default, stddev equal in x and y    
            stack = convolve(stack, kernel[None])
        ratemaps.append(stack)
    return np.array(ratemaps)

def correlate_population_vectors_unitwise(stack1, stack2):
    """Correlate two stacks of ratemaps unitwise
    Args:
        stack1 (np.ndarray): array of ratemaps, of shape (Nc, binx, biny)
        stack2 (np.ndarray): array of ratemaps, of shape (Nc, binx, biny)
    Returns:
        np.ndarray: correlations. Of shape (Nc,)
    """

    correlations = np.zeros((len(stack1), len(stack2)))
    # compute correlations for each bin
    for i in range(len(stack1)):
        for j in range(i, len(stack2)): # corrcoef is symmetric
            correlations[i,j] = np.corrcoef(stack1[i].ravel(), stack2[j].ravel())[0,1]
    return correlations

def spatial_information(s, r, bins):
    """Compute spatial information for a signal s along a trajectory specified by x and y
    Args:
        s (np.ndarray): Signal used for SI computation. Of shape (T, Nc)
            T is the length of the signal, Nc the number of units.
        r (np.ndarray): Array of coordinates, of shape (T, 2)
        bins: Bins used for ratemap construction. See binned_statistic_2d docs.
            typically tuple of ints (binx, biny), or vectors of length binx, biny. 
    Returns:
        np.array : spatial information, of shape (Nc).
    """
    # compute mean firing rate in a given bin
    x = r[:,0]
    y = r[:,1]
    mean_bin_rate, _,_,_ = scipy.stats.binned_statistic_2d(x, y, s.T, bins = bins, statistic = "mean")
    mean_rate = np.nanmean(s, axis = 0)

    # and number of visits to a bin
    counts, _,_,_= scipy.stats.binned_statistic_2d(x, y, s.T, bins = bins, statistic = "count")
    occupancy = counts/len(s) # "probability" of being in a given bin at a given time
    
    occupancy[np.isnan(occupancy)] = 0 # unvisited location do not contribute
    # sum over bin axes
    spatial_information = np.nansum(occupancy*mean_bin_rate*np.log2(mean_bin_rate/mean_rate[:,None,None]), axis = (1, 2))
    return spatial_information

def rate_overlap(stack_a, stack_b):
    """Compute rate overlap between stacks of ratemaps

    Following Leutgeb et al., the overlap 
    is measured by taking the ratio of the mean firing rate
    in the least active environment, to that in the most active.

    Args:
        stack_a (np.ndarray): stack of ratemaps, of shape (Nc, xbins, ybins),
        where N is the number of ratemaps, and xbins, ybins are the 
        number of bins in either dimension
        stack_b (np.ndarray): stack of ratemaps, of shape (N, xbins, ybins)
    Returns:
        mean_overlap (np.array): array of meaned overlaps, of shape (N,).
        std_overlap (np.array): array of standard error of the mean
        n_active (float): number of cells active in either environment
    """

    mean_a = np.nanmean(stack_a, axis = (1, 2))
    mean_b = np.nanmean(stack_b, axis = (1, 2))
    largest = np.maximum(mean_a, mean_b)
    smallest = np.minimum(mean_a, mean_b)

    rate_overlap = smallest/largest 
    return rate_overlap

def random_rate_overlap(stack_a, stack_b, iterations = 1000):
    n_cells = len(stack_a)
    random_overlap = np.zeros((iterations, n_cells))
    all_inds = np.arange(n_cells)
    available_inds = [all_inds[all_inds != i] for i in range(n_cells)]
    
    for i in range(iterations):
        inds = np.array([np.random.choice(inds) for inds in available_inds])
        random_overlap[i] = rate_overlap(stack_a, stack_b[inds])
    return random_overlap

def rate_difference(stack_a, stack_b):
    mean_a = np.nansum(stack_a, axis = (1, 2))
    mean_b = np.nansum(stack_b, axis = (1, 2))
    rate_diff = (mean_a - mean_b)/(mean_a + mean_b)
    return rate_diff

def random_rate_difference(stack_a, stack_b, iterations = 1000):
    n_cells = len(stack_a)
    random_difference = np.zeros((iterations, n_cells))
    all_inds = np.arange(n_cells)
    available_inds = [all_inds[all_inds != i] for i in range(n_cells)]
    
    for i in range(iterations):
        inds = np.array([np.random.choice(inds) for inds in available_inds])
        random_difference[i] = rate_difference(stack_a, stack_b[inds])
    return random_difference