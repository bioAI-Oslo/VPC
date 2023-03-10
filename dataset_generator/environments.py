import numpy as np

class Environments:
    def __init__(self) -> None:
        self.envs = {}
        # rotate first polygons for prettier plotting later
        self.envs["square"] = self.polygon(4) @ self.rotation_matrix(np.pi / 4)*np.sqrt(2)
        self.envs["circle"] = self.polygon(50)  # approximate circle in sad way :(
        self.envs["large_square"] = self.envs["square"] * 1.5
        self.envs["rectangle"] = self.envs["square"] * np.array([1.5, 1])[None, None]
        center_wall = np.array([[[0, -0.5], [0, 0.5]]]) # add center dividing wall to square 
        # add two superimposed walls to not break spawn algorithm
        self.envs["walled_square"] = np.r_[self.envs["square"], center_wall, center_wall] 
        self.envs["square_toroid"] = np.r_[self.envs["square"], 0.5 * self.envs["square"]]

    def rotation_matrix(self, theta):
        a = np.array([np.cos(theta), -np.sin(theta)])
        b = np.array([np.sin(theta), np.cos(theta)])
        return np.stack((a, b), axis=0)

    def polygon(self, segments):
        # create some simple geometries consisting of regular polygons
        walls = np.zeros((segments, 2, 2))
        for i in range(1, segments + 1):
            for j in range(2):
                # j index ensures that end of one segment is start of next
                walls[i - 1, j, 0] = np.cos((i + j) * 2 * np.pi / segments)
                walls[i - 1, j, 1] = np.sin((i + j) * 2 * np.pi / segments)
        return walls

    def encoding(self, name):
        # one hot encoding for environments
        ind = list(self.envs).index(name)
        return np.arange(len(self)) == ind

    def name(self, encoding):
        # get key of chamber from encoding
        ind = np.nonzero(encoding)[0][0]
        key = list(self.envs)[ind]
        return key

    def __len__(self):
        return len(self.envs)

if __name__ == "__main__":
    
    """
    import matplotlib.pyplot as plt

    environment = Environments()
    boxes = list(environment.envs)
    n = len(environment.envs)

    cols = 3
    rows = n // cols

    fig, ax = plt.subplots(rows, cols)
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count < n:
                current_box = boxes[count]
                for wall in environment.envs[current_box]:
                    ax[i, j].plot(wall[:, 0], wall[:, 1])
                    s = np.linalg.norm(wall[0] - wall[1])
                count += 1
            else:
                pass
            ax[i, j].axis("square")
            ax[i, j].axis("off")
    plt.show()

    # plot all environments superimposed to check alignment and size
    for name in environment.envs:
        # side lengths
        for wall in environment.envs[name]:
            plt.plot(wall[:, 0], wall[:, 1])
    plt.axis("square")
    plt.show()
    """