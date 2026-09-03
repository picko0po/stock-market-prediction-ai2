import torch
from torch import nn


class LSTMStocksModule(nn.Module):

    def init(self, input_size=1, hidden_size=64, num_layers=2):
        super().init()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        # x: batch, sequence
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        output, _ = self.lstm(x)

        # Last timestep
        last_output = output[:, -1, :]

        return self.fc(last_output).squeeze(-1)
