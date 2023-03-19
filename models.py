import torch
import numpy as np
from train_tools import l1_reg, l2_reg

class VPC_RNN(torch.nn.Module):
    """ RNN model for the variational position reconstruction task
    """
    def __init__(self, params, device = None, **kwargs):
        """
        Args:
            params (dict): dictionary of model parameters
            device (str, optional): device to send torch tensors to. 
                Typically "cuda" for training, and "cpu" for inference.
                Defaults to None.
        """
        super().__init__(**kwargs)
        self.params = params

        # Use cuda if available
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        n_in = int(2 + params["context"] * 6) # 6 possible contexts, 2 velocities
                
        # initialize layers
        self.g = torch.nn.RNN(
            input_size = n_in,
            hidden_size = params["nodes"],
            nonlinearity= "relu",
            bias=False,
            batch_first=True, device = self.device)
        torch.nn.init.eye_(self.g.weight_hh_l0) # identity initialization
        
        self.p = torch.nn.Linear(params["nodes"], params["outputs"], bias=False, device = self.device)
        self.eps = 1e-16 # small epsilon for center of mass estimate

        self.activation = torch.nn.ReLU()
        self.loss_fn = torch.nn.MSELoss() 

        self.to(device) 
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params["lr"])
   
    def weight_reg(self, l2):
        return l2_reg(l2, self.g.weight_hh_l0)

    def reset_state(self, shape):
        return torch.zeros(shape, device = self.device)

    def decode_phases(self, p, r):
        pt = p/(torch.sum(p, dim = -2, keepdim = True) + self.eps)
        mu = torch.sum(r[...,None, :]*pt[...,None], dim = 1) 
        return mu 

    def decode_position(self, p, mu):
        po = p/(torch.sum(p, dim = -1, keepdim = True) + self.eps)
        rhat = torch.sum(mu[:,None]*po[...,None], dim = -2)
        return rhat 

    def forward(self, inputs, g_prev=None):

        v = inputs[0] # input signal
        r = inputs[1] # true position
        
        if g_prev is None:
            initial_state = self.reset_state((v.shape[0], self.params["nodes"]))
        else:
            initial_state = g_prev
        g, _ = self.g(v, initial_state[None])
        p = self.activation(self.p(g)) 
        
        mu = self.decode_phases(p, r)
        yhat = self.decode_position(p, mu)
        return yhat, g, p, mu

    def train_step(self, x, y, g_prev = None):
        self.optimizer.zero_grad(set_to_none=True)        
        yhat, g, p, mu = self(x, g_prev)
        weight_reg = self.weight_reg(self.params["l2"])
        activity_reg = l1_reg(self.params["al1"], g)
        loss = self.loss_fn(yhat, y) + weight_reg + activity_reg
        
        # parameter update
        loss.backward()
        self.optimizer.step()
        return loss, yhat, g

    def val_step(self, x, y, g_prev = None):
        # val step and train step are equal, except for gradient
        with torch.no_grad():
            yhat, g, p, mu = self(x, g_prev)
            weight_reg = self.weight_reg(self.params["l2"])
            activity_reg = l1_reg(self.params["al1"], g)
            val_loss = self.loss_fn(yhat, y) + weight_reg + activity_reg
        return val_loss, yhat, g
    
    def inference(self, dataset):
        """Run model in inference mode, returning metrics *and* states
        Args:
            dataset : iterable of length N, returning input output pairs of torch tensors
            should contain tensors of ((v, r), r) with shape 
            (((BS, T, Nin), (BS, T, 2)), (BS, T, 2)),
            where BS is the batch size, T the number of timesteps
            and Nin the number of inputs to the recurrent layer 
            (velocities + optional border + context signals)
        Returns:
            gs (np.ndarray): Recurrent states for each sample in dataset, shape (N, BS, T, Ng)
            ps (np.ndarray): Output states for each sample in dataset, shape (N, BS, T, Np)
            centers (np.ndarray): Center estimate for each sample in dataset, shape (N, Bs, Np, 2)
            preds (np.ndarray): Model predictions for each sample in dataset, shape (N, BS, T, 2)
            metrics (dict): Contains lists of inference metrics for each sample. 
        """
        
        gs = []
        ps = []
        centers = []
        preds = []

        with torch.no_grad():
            rnn_state = None # sample initial state
            for i, (x, y_true) in enumerate(dataset):

                reset_state = (i % self.params["reset_interval"]) == 0
                if reset_state:
                    rnn_state = None
                else:
                    rnn_state = g[:, -1].detach().clone().to(self.device)

                y_pred, g, p, center = self(x, rnn_state)

                gs.append(g.cpu().numpy())
                ps.append(p.cpu().numpy())
                centers.append(center.cpu().numpy())
                preds.append(y_pred.cpu().numpy())

        # concat in same order
        return [np.concatenate(var, axis = 0) for var in [gs, ps, centers, preds]]
    
class VPC_FF(VPC_RNN):
    """ Feed Forward deep network for the variational position reconstruction task
    """
    def __init__(self, params, device = None, **kwargs):
        """
        Args:
            params (dict): dictionary of model parameters
            device (str, optional): device to send torch tensors to. 
                Typically "cuda" for training, and "cpu" for inference.
                Defaults to None.
        """
        super().__init__(params, device, **kwargs)
        n_in = int(2 + params["context"] * 6) # 6 possible contexts, 2 velocities

        self.g = torch.nn.Sequential(
            torch.nn.Linear(n_in, 64, bias = False),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128, bias = False),
            torch.nn.ReLU(),
            torch.nn.Linear(128, params["nodes"], bias = False),
            torch.nn.ReLU())
        
        self.p = torch.nn.Linear(params["nodes"], params["outputs"], bias = False)
        self.to(self.device) 
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params["lr"])
    
    def weight_reg(self, l2):
        return l2_reg(l2, self.g[-2].weight[-1])
        
    def forward(self, inputs, g_prev = None):
        rc = inputs[0]
        r = inputs[1]
        g = self.g(rc)
        p = self.activation(self.p(g))
        mu = self.decode_phases(p, r)
        yhat = self.decode_position(p, mu)
        return yhat, g, p, mu