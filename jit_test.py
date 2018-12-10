import torch

def sin(x):
    return torch.sin(x, casting='no')

traced_sin = torch.jit.trace(sin, (torch.rand(3),))

print(traced_sin(torch.rand(5)))
