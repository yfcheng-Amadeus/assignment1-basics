import einx
import torch
import math
from jaxtyping import Float, Int
from torch import Tensor

from .nn_utils import silu

class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return einx.dot("d_out d_in, ... d_in -> ... d_out", self.weight, input)

class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.weight[input]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model:int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.empty((d_model,), device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_sum_squared = einx.sum("... d_model -> ... 1", x**2, keepdim=True)
        rms = torch.sqrt(x_sum_squared / self.d_model + self.eps)
        return einx.multiply("... d_model, ... d_model -> ... d_model", self.weight / rms, x)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = torch.nn.Parameter(
            torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype)
            )
        self.w2 = torch.nn.Parameter(
            torch.empty(self.d_model, self.d_ff, device=device, dtype=dtype)
            )
        self.w3 = torch.nn.Parameter(
            torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype)
            )

    def forward(self, x: Float[Tensor, "... d_model"])-> Float[Tensor, "... d_model"]:
        x1 = einx.dot("... d_model, d_ff d_model -> ... d_ff", x, self.w1)
        silu_x1 = silu(x1)
        x3 = einx.dot("... d_model, d_ff d_model -> ... d_ff", x, self.w3)
        x2 = einx.multiply("... d_ff, ... d_ff -> ... d_ff", silu_x1, x3)
        return einx.dot("... d_ff, d_model d_ff -> ... d_model", x2, self.w2)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryPositionalEmbeddings(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        pos = torch.arange(self.max_seq_len, dtype=torch.int, device=self.device)

        theta = 1.0/ torch.pow(self.theta, torch.arange(0, self.d_k, 2, dtype=torch.float) / self.d_k)
        idx_theta = einx.multiply("... s, ... d -> s d", pos, theta)
        cache = torch.stack(
            (torch.cos(idx_theta), torch.sin(idx_theta)), dim=-1
        )
        
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... sequence_length d_k"], 
        token_positions: Int[Tensor, "... sequence_length"]
    ) -> torch.Tensor:
       
        if token_positions.ndim >=1 :
            token_positions = token_positions.squeeze(0)
       
        *batch_dims, sequence_length, d_k = x.shape
        
        rope_cache = self.cache[token_positions]
        

        # shape [n, s, d_k] -> [n, s , d_k // 2, 2]
        x = einx.rearrange("... s (d k) -> ... s d k", x, d=self.d_k // 2)
        #print(f"rope_cache shape: {rope_cache.shape}, x shape: {x.shape}")
        x_out = torch.stack(
            (
                einx.multiply("... s d,  s d -> ... s d", x[..., 0], rope_cache[..., 0]) - einx.multiply("... s d,  s d -> ... s d", x[..., 1], rope_cache[..., 1]),
                einx.multiply("... s d,  s d -> ... s d", x[..., 1], rope_cache[..., 0]) + einx.multiply("... s d,  s d -> ... s d", x[..., 0], rope_cache[..., 1])
            ),
            dim=-1,
        )
        #print(f"rope_cache shape: {rope_cache.shape}, x_out shape: {x_out.shape}")
        return einx.rearrange("... s d k -> ... s (d k)", x_out, d=self.d_k // 2)

