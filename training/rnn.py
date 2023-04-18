import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, input_size, dropout_rate=0.5):
        super().__init__()
        self.rnn = nn.LSTM(input_size, 256, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64,1)
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, x):
        x = x.squeeze(1)
        print(x.shape)
        x = self.layer_norm(x)
        x, (hidden_last, cell_last) = self.rnn(x)
        hidden_last = self.dropout(hidden_last)

        hidden_last = hidden_last.squeeze(0)
        x = self.classifier(hidden_last)
        return x