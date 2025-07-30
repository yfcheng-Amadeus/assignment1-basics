import torch
from jaxtyping import Float, Int
from torch import Tensor
import einx
import numpy as np

class Softmax(torch.nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Float[Tensor, "..."])-> Float[Tensor, "..."]:
        max_values = x.max(dim=self.dim).values
        x_standardized = einx.subtract("... s, ... -> ... s", x, max_values)
        x_exp = torch.exp(x_standardized)
        x_exp_sum = einx.sum("... s -> ...", x_exp, dim=self.dim, keepdim=True)
        return einx.divide("... s, ...  -> ... s", x_exp, x_exp_sum)

def sigmoid(x: Float[Tensor, "..."])-> Float[Tensor, "..."]:
    return torch.exp(x) / (1 + torch.exp(x))

def silu(x: Float[Tensor, "..."])-> Float[Tensor, "..."]:
    return x * sigmoid(x)


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
    
    def forward(
        self,
        inputs: Float[Tensor, "batch_size vocab_size"],
        targets: Int[Tensor, "batch_size"]
    ):  
        batch_size, vocab_size = inputs.shape
        inputs = einx.subtract("batch_size s, batch_size -> batch_size s", inputs, inputs.max(dim=-1).values)
        log_sum_exp = einx.logsumexp("batch_size s -> batch_size", inputs, dim=-1)
        select_inputs = einx.get_at('b [n], b -> b', inputs, targets)
        loss = einx.subtract("b, b -> b", select_inputs, log_sum_exp)
        loss = einx.sum("b -> ", loss, dim=-1)
        loss = -loss / batch_size
        return loss
        
        
        










        




  

        
    





        