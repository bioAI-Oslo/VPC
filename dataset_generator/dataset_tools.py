import numpy as np

def rejection_sample(target_shape, proposal_dist, criterion):
    """
    All-purpose method for doing rejection sampling
    params:
    target_shape: Shape of the array to be generated
    proposal_dist: distribution function to draw proposed samples from.
    Must take argument samples and return samples of shape (samples, *target_shape[1:])
    criterion: function which takes in proposed samples, and returns boolean mask
    determining whether or not each input sample is accepted.
    returns:
    accepted_samples: Ndarray of accepted samples of shape target_shape
    """

    n_samples = target_shape[0]

    accepted_samples = np.zeros(target_shape)
    remaining_inds = np.arange(n_samples)

    remaining = n_samples

    while remaining > 0:
        proposed = proposal_dist(remaining, remaining_inds)

        accept_mask = criterion(proposed, remaining_inds)  # check for acceptance
        accepted_inds = remaining_inds[accept_mask] 
        accepted_samples[accepted_inds] = proposed[accept_mask]  # update accepted samples by inds

        remaining_inds = remaining_inds[~accept_mask]  # remove accepted indices
        remaining = len(remaining_inds)  # update number of remaining samples

    return accepted_samples

def bounding_rectangle(walls):
    # find smallest bounding rectangle that encapsulates all points in walls
    eps = 1e-6  # with a margin of safety 
    lower = np.amin(walls, axis=(0, 1)) * (1 + eps)
    upper = np.amax(walls, axis=(0, 1)) * (1 + eps)
    return lower, upper

def cross_2d(x, y):
    # 2d cross product, determinant, computed along last dims of x and y
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

def line_line_intersect(a, b):
    """
    Compute whether line segments in a intersect with any line segment in b.
    params:
    a, b: 2-tuples containing ndarrays of lines to be compared for intersection
    a and b must be of shape x.shape = (N, 2, 2), where N is the number of line segments,
    and x[0] is the starting point of the segment, and x[1] the endpoint.
    returns:
    intersects: Ndarray of intersections of dtype bool
    Note that the dimension of the output will follow the input; if the inputs are of shape
    a.shape = (Na, 2, 2), b.shape = (Nb, 2, 2), the return will be of shape (Na, Nb)
    """

    na = a[:, 1] - a[:, 0]  # direction vectors
    nb = b[:, 1] - b[:, 0]

    b_hat = b[None, :, 0] - a[:, None, 0]  # difference in starting points

    # compute line-line intersects
    # for each a, compute intersects with  each b: expand 0th dim of b, 1st dim of a
    normal = cross_2d(na[:, None], nb[None])
    sa = cross_2d(b_hat, nb[None]) / normal  # line parameterization
    sb = cross_2d(b_hat, na[:, None]) / normal

    # line parameter must be positive and intersect must be found on each line
    intersect = (sa > 0) * (sa < 1) * (sb > 0) * (sb < 1)
    return intersect 