import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

INF = 1e18
INIT = 1e-2

class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, mask=None):
        # query: Tensor (batch_size, query_size)
        # key: Tensor (batch_size, time_step, key_size)
        # value: Tensor (batch_size, time_step, hidden_size)
        # mask: Tensor (batch_size, time_step)
        score = self._score(query, key)
        probability = self._probability_normalize(score, mask)
        output = self._attention_aggregate(probability, value)
        return output

    def _score(self, query, key):
        raise NoImplementedError('Attention score method is not implemented.')

    def _probability_normalize(self, score, mask):
        if mask is not None:
            score = score.masked_fill(mask.unsqueeze(1) == 0, -INF)
        probability = F.softmax(score, dim=-1)
        return probability

    def _attention_aggregate(self, probability, value):
        return probability.matmul(value).squeeze(1)

    def get_attention_weights(self, query, key, mask=None):
        score = self._score(query, key)
        probability = self._probability_normalize(score, mask)
        return probability

class DotAttention(Attention):

    def __init__(self):
        super(DotAttention, self).__init__()

    def _score(self, query, key):
        assert query.size(1) == key.size(2)
        return query.unsqueeze(1).matmul(key.transpose(1, 2))

class ScaledDotAttention(Attention):

    def __init__(self):
        super(ScaledDotAttention, self).__init__()

    def _score(self, query, key):
        assert query.size(1) == key.size(2)
        return query.unsqueeze(1).matmul(key.transpose(1, 2)) / math.sqrt(query.size(1))

class AdditiveAttention(Attention):

    def __init__(self, query_size, key_size):
        super(AdditiveAttention, self).__init__()
        self._projection = nn.Linear(query_size + key_size, 1)

    def _score(self, query, key):
        time_step = key.size(1)
        query = query.repeat(time_step, 1, 1).transpose(0, 1)  # (batch_size, time_step, query_size)
        scores = self._projection(torch.cat([query, key], dim=2)).transpose(1, 2)
        # scores = torch.tanh(scores)
        return scores

class MultiplicativeAttention(Attention):

    def __init__(self, query_size, key_size):
        super(MultiplicativeAttention, self).__init__()
        self._weights = nn.Parameter(torch.Tensor(key_size, query_size))
        init.uniform_(self._weights, -INIT, INIT)

    def _score(self, query, key):
        batch_size = query.size(0)
        time_step = key.size(1)
        weights = self._weights.repeat(batch_size, 1, 1)  # (batch_size, key_size, query_size)
        query = query.unsqueeze(-1)  # (batch_size, query_size, 1)
        mids = weights.matmul(query)  # (batch_size, key_size, 1)
        mids = mids.repeat(time_step, 1, 1, 1).transpose(0, 1)  # (batch_size, time_step, key_size, 1)
        key = key.unsqueeze(-2)  # (batch_size, time_step, 1, key_size)
        scores = key.matmul(mids).squeeze(-1).transpose(1, 2)  # (batch_size, time_step)
        # scores = torch.tanh(scores)
        return scores

class MultiLayerPerceptronAttention(Attention):

    def __init__(self, query_size, key_size, out_size=1):
        super(MultiLayerPerceptronAttention, self).__init__()
        self._layer1 = nn.Linear(query_size + key_size, out_size, bias=False)
        self._layer2 = nn.Linear(out_size, 1, bias=False)

    def score(self, query, key):
        time_step = key.size(1)
        query = query.repeat(time_step, 1, 1).transpose(0, 1)  # (batch_size, time_step, query_size)
        score = self._layer2(torch.tanh(self._layer1(torch.cat([query, key], dim=2)))).transpose(1, 2)
        return score