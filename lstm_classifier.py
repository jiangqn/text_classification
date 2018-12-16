import torch
import torch.nn as nn

class LstmClassifier(nn.Module):
    
    def __init__(self, vocab_size, class_num=2, embed_dim=128, hidden_size=128):
        super(LstmClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, class_num)
    
    def forward(self, x):
        x = self.embedding(x)
        lstm_output, _ = self.lstm(x)
        output = lstm_output.mean(dim=1)
        output = self.fc(output)
        return output