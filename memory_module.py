import torch
import torch.nn as nn
from torch.nn import init
from attention import *

class Memory(nn.Module):

    def __init__(self, memory_size, feature_size):
        super(Memory, self).__init__()
        self._memory = nn.Parameter(
            torch.Tensor(memory_size, feature_size)
        )
        # init.uniform_(self._memory, -INIT, INIT)
        init.xavier_uniform_(self._memory)
        self._attn = ScaledDotAttention()

    def forward(self, x):
        memory = self._memory.repeat(x.size(0), 1, 1)
        return x + self._attn(x, memory, memory)