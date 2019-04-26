import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):

    def __init__(self, vocab_size, kernels=[[100, 1], [100, 2], [100, 3], [100, 4], [100, 5]], dropout=0.5, class_num=2, embed_dim=128):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv_layers = nn.ModuleList()
        fc_input_dim = 0
        for kernel in kernels:
            self.conv_layers.append(nn.Conv2d(1, kernel[0], (kernel[1], embed_dim)))
            fc_input_dim += kernel[0]
        self.memory = Memory(50, fc_input_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_input_dim, class_num)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        fc_input_list = []
        for conv in self.conv_layers:
            y = conv(x)
            y = y.squeeze(3)
            y = F.relu(y)
            y = F.max_pool1d(y, y.size(2)).squeeze(2)
            fc_input_list.append(y)
        fc_input = torch.cat(fc_input_list, 1)
        fc_input = self.memory(fc_input)
        fc_input = self.dropout(fc_input)
        output = self.fc(fc_input)
        return output