import torch
import numpy as np
import numpy.typing as npt
import os
import typing

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str = "cpu",
):
    indices = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    x = torch.tensor([dataset[i:i + context_length] for i in indices], device=device)
    y = torch.tensor([dataset[i + 1:i + 1 + context_length] for i in indices], device=device)
    return x, y

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):  
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint.get('iteration', 0)
    return iteration

    