import numpy as np
from scipy import stats

import dataset_tools

class DataGenerator:
    def __init__(self) -> None:
        """ Class for generating trajectories
        """
        self._kappa_hd = 4  # sharpness of hd sampling
        self._std_s = 0.5  # width of speed sampling pdf
        self.dt = 0.1

    def spawn(self, samples, walls):
        """
        Spawn samples 2D (points) inside geometries given by line segments in walls, by ray marching.
        params:
        samples: int, number of samples to generate
        chambers: ndarray, of shape (N_walls, 2, 2), where N_walls is the number of walls that
                  make up the geometry.
        """
        # spawn by doing rejection sampling in box
        lower, upper = dataset_tools.bounding_rectangle(walls)
        proposer = lambda samples, _: np.random.uniform(lower, upper, (samples, 2))

        def inside_bounds(x, rem):
            a = x  # line segment starting at x
            b = x + np.array([1e6, 1e-6])  # end far right of box; tiny skew to avoid being wall-parallel
            line = np.stack((a, b), axis=1)  # line segment of shape (samples, 2, 2)
            # determine number of intersects
            intersect = dataset_tools.line_line_intersect(line, walls)
            n_intersects = np.count_nonzero(intersect, axis=-1)
            # if number of intersects is not even, it is inside!
            inside = np.mod(n_intersects, 2) != 0
            return inside

        spawned_points = dataset_tools.rejection_sample((samples, 2), proposer, inside_bounds)
        return spawned_points

    def step(self, r, hd, walls, dt = 1):
        """
        Take a single step in 2D geometry defined by walls, starting
        from r, with previous heading hd
        """
        # new head direction is drawn from distribution centered at previous hd
        samples = len(r)
        # draw all step sizes beforehand
        s = np.random.rayleigh(self._std_s, samples)

        # step distribution: Propose head direction and resulting step
        def step_proposer(samples, rem):
            theta_prop = stats.vonmises.rvs(self._kappa_hd, hd[rem], size=(samples))
            x_prop = s[rem]*np.cos(theta_prop)
            y_prop = s[rem]*np.sin(theta_prop)
            prop_params = np.stack((x_prop, y_prop, theta_prop), axis=-1)
            return prop_params

        def is_allowed(proposed_move, rem):
            a = r[rem]  # only check remaining samples
            b = r[rem] + dt*proposed_move[:, :2]  # coordinate step is first 2 components
            step_line = np.stack((a, b), axis=1)  # stack into array of line segments
            # check for any intersections
            intersect = dataset_tools.line_line_intersect(step_line, walls)
            # only intersection-free moves are allowed
            allowed = np.count_nonzero(intersect, axis=-1) == 0
            return allowed

        # propose head direction and step in one array
        accepted_steps = dataset_tools.rejection_sample(np.array([samples, 3]), step_proposer, is_allowed)
        return accepted_steps

    def generate_paths(self, samples, timesteps, walls):
        r = np.zeros((samples, timesteps, 2))  # all positions
        hd = np.zeros((samples, timesteps))  # all head directions
        v = np.zeros((samples, timesteps - 1, 2))  # all velocities

        # initial conditions
        r[:, 0] = self.spawn(samples, walls) # spawn points inside geometry
        hd[:, 0] = np.random.uniform(0, 2 * np.pi, (samples)) # initial heading

        for i in range(timesteps - 1):
            # step is packaged array of velocity and sampled head direction
            step = self.step(r[:, i], hd[:, i], walls, self.dt)
            v[:, i] = step[:, :2]
            hd[:, i + 1] = step[:, -1]
            r[:, i + 1] = r[:, i] + self.dt*v[:, i]
        return r, v

    def generate_points(self, samples, space_samples, walls):
        r = np.zeros((samples, space_samples, 2))  # all positions

        for i in range(space_samples):
            r[:,i] = self.spawn(samples, walls)
        return r

