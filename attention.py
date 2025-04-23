import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim)
        self.Ua = nn.Linear(hidden_dim, hidden_dim)
        self.Va = nn.Linear(hidden_dim, 1)
        
    def forward(self, query, keys):

        scores = self.Va(torch.tanh(self.Wa(query).unsqueeze(1) + self.Ua(keys)))
            
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.transpose(1, 2), keys)
        weights = weights.squeeze(2)
        
        return context, weights
    