Code for reproducing results in [Decoding the Cognitive map: Learning place cells and remapping](https://elifesciences.org/reviewed-preprints/99302), published in eLife (2024).

To train models and generate results, first generate a dataset, using the [create_dataset.ipynb](dataset_generator/create_dataset.ipynb) notebook. The dataset contains 500-timestep trajectories visiting six distinct geometries. Network inputs consist of velocities along such trajectories, alongside a time-constant context signal unique to each environment. Labels contain Cartesian coordinates along trajectories. For non-path integrating models, uniformly sampled datasets may also be created, where both labels and inputs are Cartesian coordinates.

Then, create an experiment (i.e., a model) by running [model_setup.py](model_setup.py). A model name and path can be passed as the first argument to this run. This creates a model directory, wherein a JSON file is created, which specifies model hyperparameters. Edit this file to change e.g. the number of recurrent units. 

Finally, train a model by running [train_rnn.ipynb](train_rnn.ipynb) (for trajectory data) to train a recurrent network. Subsequent analyses can be found in the [notebooks](notebooks) directory. For example, running [spatial_representations.ipynb](notebooks/spatial_representations.ipynb) allows for loading a model, running it on a test dataset, and computing ratemaps of unit responses. 

