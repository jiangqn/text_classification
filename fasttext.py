import torch
import torch.nn as nn

class FastText(nn.Module):
    
    def __init__(self, vocab_size, class_num=2, embed_dim=128):
        super(FastText, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, class_num)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x