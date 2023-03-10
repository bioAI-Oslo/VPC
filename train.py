import numpy as np
import torch

from tools import Logger, DatasetLoader, l1_reg, l2_reg, euclid
from tqdm import tqdm

def validate(model, dataset):
    """
    Run network on validation dataset.
    Args:
        dataset (DatasetLoader): iterable dataset, returning
            input/output pairs
    Returns:
        dict: validation metrics
    """
    # compute the same as for training, but on validation set
    metrics = {"al1":0, "mse":0, "euclid":0}  
    with torch.no_grad():
        for i, (x, y_true) in enumerate(dataset):
            # loop same as for training, except for gradients
            reset_state = (i % model.params["reset_interval"]) == 0
            if reset_state:
                initial_rnn_state = None
            else:
                initial_rnn_state = g[:, -1].detach().clone().to(model.device)
                        
            y_pred, g, _, _ = model(x, initial_rnn_state)

            # keep running average
            metrics["al1"] += l1_reg(model.params["al1"], g).item()
            metrics["euclid"] += euclid(y_pred, y_true).item()
            metrics["mse"] += model.mse(y_pred, y_true).item()
    
    # average over all batches
    metrics = {metric: metrics[metric]/len(dataset) for metric in metrics}
    return metrics

def train(model, optimizer, train_path, val_path, save_dir):
    """Train a model
    Args:

        optimizer (torch.optim.Optimizer): Some torch optimizer, typically Adam
        save_path (str): path to directory in which to save model 
    """
    # set up logger to save metrics
    logger = Logger(save_dir)
    train_bar = tqdm(range(model.params["epochs"]))

    # outer training loop
    for epoch in train_bar:
        # set up dataset loaders
        train_dataset = DatasetLoader(train_path, model.params, model.device)
        val_dataset = DatasetLoader(val_path, model.params, model.device)
        # train metrics are reset every epoch
        train_metrics = {"al1":0, "l2":0, "mse":0, "euclid":0}
        # inner training loop
        for i, (x_train, y_train) in enumerate(train_dataset):
            # reset gradients
            optimizer.zero_grad(set_to_none=True)
            
            reset_state = (i % model.params["reset_interval"]) == 0
            if reset_state:
                initial_rnn_state = None # resample state
            else:
                initial_rnn_state = g[:, -1].detach().clone().to(model.device) # persistent RNN state

            position_estimate, g, p, centers = model(x_train, initial_rnn_state)

            # reconstruction error MSE
            position_loss = (position_estimate - y_train)**2
            mse = torch.mean(position_loss)
            #mse = model.mse(position_estimate, y_train) # position loss
            al1 = l1_reg(model.params["al1"], g) # regularization 
            wl2 = l2_reg(model.params["l2"], model.rnn_layer.weight_hh_l0)
            loss = mse + al1 + wl2 # total loss

            loss.backward() # update gradients
            optimizer.step()

            # training metrics
            with torch.no_grad():
                euclid_error = euclid(position_estimate, y_train).item()
                # update training metric running average
                train_metrics["mse"] += mse.item()
                train_metrics["al1"] += al1.item() 
                train_metrics["l2"] += wl2.item() 
                train_metrics["euclid"] += euclid_error
                # if nan loss, stop training
                nan_detected = np.isnan(euclid_error)
                if nan_detected:
                    break

        if nan_detected:
                print(f"NaN euclid loss detected at epoch {epoch}, ending training...")
                break

        # average over samples
        train_metrics = {metric: train_metrics[metric]/len(train_dataset) for metric in train_metrics}
        logger(train_metrics, "train")

        # Validation step
        val_metrics = validate(model, val_dataset) # compute validation metrics
        logger(val_metrics, "val") # log metrics
        train_bar.set_description(f"Val MERE {val_metrics['euclid']:.3f}")
        #train_bar.update()

    # Save loss/metrics history
    logger.save_loss() 
    
    
    
dataset = TensorDataset(torch.tensor(x), torch.tensor(y, dtype = torch.long))
train_data, val_data, test_data = random_split(dataset, [train_ratio, validation_ratio, test_ratio]) 

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last = True)
val_loader = DataLoader(val_data, batch_size=batch_size, drop_last = True)
