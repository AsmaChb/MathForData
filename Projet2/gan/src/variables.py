import torch

from torch import nn
  

class Generator(nn.Module):
    def __init__(self, noise_dim=2, output_dim=2, hidden_dim=100):
        super().__init__()
        
        self.linear1 = nn.Linear(noise_dim, hidden_dim )
        self.linear2 = nn.Linear(hidden_dim, output_dim )
        
        self.inner = nn.Sequential(self.linear1, nn.ReLU(), self.linear2)
        
 

    def forward(self, z):
        """
        Evaluate on a sample. The variable z contains one sample per row
        """
        return self.inner(z)


class DualVariable(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=100, c=1e-2):
        super().__init__()
        self.c=c
        
        self.linear1 = nn.Linear(input_dim, hidden_dim )
        self.linear2 = nn.Linear(hidden_dim, 1 )
        
        self.inner = nn.Sequential(self.linear1, nn.ReLU(), self.linear2)
                      
        

    def forward(self, x):
        """
        Evaluate on a sample. The variable x contains one sample per row
        """
        return self.inner(x)

    def enforce_lipschitz(self):
        """Enforce the 1-Lipschitz condition of the function by doing weight clipping"""
        self.weight_clipping()

    def weight_clipping(self):
        """
        Clip the parameters to $-c,c$. You can access a modules parameters via self.parameters().
        Remember to access the parameters  in-place and outside of the autograd with Tensor.data.
        """
        with torch.no_grad():
            for p in self.parameters():
                p.data  = torch.clip(p.data,-self.c,self.c)
                
                    

