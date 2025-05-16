import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self,
        n_input_channels,
        n_output_channels,
        height=48,
        width=72,
        hidden_size=1024,
        dropout=0.2,
    ):
        super().__init__()
        input_size = n_input_channels * width * height
        output_size = n_output_channels * width * height

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.batchNorm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
        self.n_output_channels = int(n_output_channels)
        self.height = height
        self.width = width

    def forward(self, x):
        x = self.flatten(x)   
        x = self.linear1(x)
        x = self.batchNorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x) 

        x = x.unflatten(1, (self.n_output_channels, self.height, self.width))
        return x