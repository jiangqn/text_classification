import torch
import torch.nn as nn
from memory_module import Memory

class FastText(nn.Module):
    
    def __init__(self, vocab_size, class_num=2, embed_dim=128):
        super(FastText, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, class_num)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)
        x = self.fc2(x)
        x = self.fc(x)
        return x