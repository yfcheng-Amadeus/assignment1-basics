import einx
import torch
from jaxtyping import Float, Int
from torch import Tensor

from .layers import RotaryPositionalEmbeddings
from .nn_utils import Softmax

class scaled_dot_product_attention(torch.nn.Module):
    def __init__(self, d_k: int, device=None, dtype=None):
        super().__init__()
        self.d_k = d_k
        self.scale = 1.0 / (d_k ** 0.5)
        self.device = device
        self.dtype = dtype

    def forward(
        self,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Float[Tensor, " ... queries keys"] | None = None
    ) -> Float[Tensor, " ... queries d_v"]:
        scores = einx.dot("... queries d_k, ... keys d_k -> ... queries keys", Q, K) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        softmax = Softmax(dim=-1)
        attn_weights = softmax(scores)
        return einx.dot("... queries keys, ... keys d_v -> ... queries d_v", attn_weights, V)

class MultiheadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.Q_proj = torch.nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )
        self.K_proj = torch.nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )
        self.V_proj = torch.nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )
        self.out_proj = torch.nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )
        self.attention = scaled_dot_product_attention(d_k=self.d_head, device=device, dtype=dtype)

    def forward(
        self,
        in_features: Float[Tensor, "... sequence_length d_in"],
    ) -> Float[Tensor, "... sequence_length d_out"]:
        *batch_dims, sequence_length, d_model = in_features.shape

        Q = einx.dot(("... s d_model, d d_model -> ... s d "), in_features, self.Q_proj)
        K = einx.dot(("... s d_model, d d_model -> ... s d "), in_features, self.K_proj)
        V = einx.dot(("... s d_model, d d_model -> ... s d "), in_features, self.V_proj)

        Q = einx.rearrange("... s (h d) -> ... h s d", Q, h=self.n_heads)
        K = einx.rearrange("... s (h d) -> ... h s d", K, h=self.n_heads)
        V = einx.rearrange("... s (h d) -> ... h s d", V, h=self.n_heads)

        casual_mask = torch.tril(
            torch.ones((sequence_length, sequence_length), device=Q.device, dtype=Q.dtype)
        )

        attn_output = self.attention(Q, K, V, mask=casual_mask)

        out = einx.dot('... h s d, d_model (h d) -> ... s d_model', attn_output, self.out_proj, h=self.n_heads)
        return out

class MultiheadSelfAttentionWithRoPE(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        theta: float,
        max_seq_len,
        device=None,
        dtype=None,
        
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0
        self.Q_proj = torch.nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )
        self.K_proj = torch.nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )
        self.V_proj = torch.nn.Parameter(   
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )
        self.out_proj = torch.nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )
        self.attention = scaled_dot_product_attention(d_k=self.d_head, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbeddings(
            theta=theta,
            d_k=d_model // n_heads,
            max_seq_len=2048,
            device=device
        )
    
    def forward(
        self,
        in_features: Float[Tensor, "... sequence_length d_in"],
        token_positions: Int[Tensor, "... sequence_length"]
    ) -> Float[Tensor, "... sequence_length d_out"]:

        *batch_dims, sequence_length, d_model = in_features.shape

        Q = einx.dot(("... s d_model, d d_model -> ... s d "), in_features, self.Q_proj)
        K = einx.dot(("... s d_model, d d_model -> ... s d "), in_features, self.K_proj)
        V = einx.dot(("... s d_model, d d_model -> ... s d "), in_features, self.V_proj)

        Q = einx.rearrange("... s (h d) -> ... h s d", Q, h=self.n_heads)
        K = einx.rearrange("... s (h d) -> ... h s d", K, h=self.n_heads)
        V = einx.rearrange("... s (h d) -> ... h s d", V, h=self.n_heads)

        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        casual_mask = torch.tril(
            torch.ones((sequence_length, sequence_length), device=Q.device, dtype=Q.dtype)
        )

        attn_output = self.attention(Q, K, V, mask=casual_mask)

        out = einx.dot('... h s d, d_model (h d) -> ... s d_model', attn_output, self.out_proj, h=self.n_heads)
        return out
        
        
        