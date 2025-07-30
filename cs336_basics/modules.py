import einx
import torch
from torch import Tensor
from jaxtyping import Float, Int

from .layers import Linear, RMSNorm, SwiGLU, Embedding
from .nn_utils import Softmax
from .attention import MultiheadSelfAttentionWithRoPE

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, max_seq_len: int, 
                 theta: float):
        super().__init__()
        self.attn = MultiheadSelfAttentionWithRoPE(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            theta=theta
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
        self.norm1 = RMSNorm(d_model=d_model)
        self.norm2 = RMSNorm(d_model=d_model)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.attn(x, token_positions)
        x = einx.add("... s d_model, ... s d_model -> ... s d_model", x, residual)
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = einx.add("... s d_model, ... s d_model -> ... s d_model", x, residual)
        return x


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.token_embedding = Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )
        
    
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model, d_ff, num_heads, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        self.softmax = Softmax(dim=-1)

    def forward(self, x: Int[Tensor, "b s"]) -> Float[Tensor, "b s vocab_size"]:
        token_embeddings = self.token_embedding(x)
        token_positions = torch.arange(
            x.shape[-1], device=x.device, dtype=torch.int32
        ).unsqueeze(0)
        for layer in self.layers:
            token_embeddings = layer(token_embeddings, token_positions)
        out = self.norm(token_embeddings)
        out = self.lm_head(out)
        #out = self.softmax(out)
        return out