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
            batch_first=True)
        torch.nn.init.eye_(self.g.weight_hh_l0) # identity initialization
        
        self.p = torch.nn.Linear(params["nodes"], params["outputs"], bias=False)
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
        # (bs, ts, 1, 2) * (bs, ts, N, 1)
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
            initial_state = g_prev.detach().clone() # persistent RNN state
        
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
        loss = self.loss_fn(yhat, y)# + weight_reg + activity_reg
        
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
            val_loss = self.loss_fn(yhat, y)# + weight_reg + activity_reg
        return val_loss, yhat, g

class VPC_FC(VPC_RNN):
    """ Fully connected deep network for the variational position reconstruction task
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
            torch.nn.Linear(n_in, 512, bias = False),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512, bias = False),
            torch.nn.ReLU(),
            torch.nn.Linear(512, params["nodes"], bias = False),
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