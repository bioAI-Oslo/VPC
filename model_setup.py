import sys, os
import json

class Parameters:
    """ Class for setting all parameters pertaining to an experiment 
    """

    def __init__(self, loc=None) -> None:
        """Set experiment parameters

        Attributes:
            loc (str, optional): location of pre-existing parameters to be loaded. If None, 
                a new dict of experiment parameters is created. Defaults to None.
        """
        if loc is None:
            # if no load path is given, use default parameters
            self.params = {}  # experimental parameters
            self.params["epochs"] = 100 
            self.params["batch_size"] = 64
            self.params["lr"] = 1e-4  # learning rate
            self.params["al1"] = 10.0  # l1 activity regularization
            self.params["l2"] = 0  # l2 weight regularization
            self.params["nodes"] = 500  # number of recurrent nodes
            self.params["outputs"] = 100  # number of output nodes
            self.params["reset_interval"] = 10  # > 1 is stateful
            self.params["context"] = True  # whether to give model context signal
        else:
            # load experimental parameters from file
            self.params = self.load_params(loc)  

    def save_params(self, path):
        """ Save class parameters to .json file

        Args:
            path (str): file location; where to store .json file
        """
        with open(f"{path}/model_parameters.json", "w") as f:
            json.dump(self.params, f, indent=4)

    def load_params(self, path):
        """ Load class parameters from .json file

        Args:
            path (str): File location of JSON model specification

        Returns:
            loaded_params (dict): experiment parameter dictionary loaded from file
        """
        # load parameters from json file
        file = f"{path}/model_parameters.json"
        with open(file, "r") as f:
            loaded_params = json.load(f)
        return loaded_params

if __name__ == "__main__":
    # Enter path to model, on the form mydir/experiment_name
    try: 
        path = sys.argv[1]
    except IndexError:
        path = "./VPC"
        print(f"No model path given. Default to {path}")

    # create directories if they do not exist
    if not os.path.exists(path):
        os.makedirs(path)

    parameters = Parameters()
    parameters.save_params(path)
