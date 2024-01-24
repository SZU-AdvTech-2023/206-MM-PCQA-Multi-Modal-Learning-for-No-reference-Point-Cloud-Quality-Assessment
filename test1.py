import torch
from torch import nn

m = nn.AdaptiveAvgPool1d(1)
input = torch.randn(6, 1024)
output = m(input)
print(output.shape)