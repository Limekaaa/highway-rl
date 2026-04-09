import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, V: int, F: int, nb_actions: int):
        """
        V: number of vehicles
        F: number of features
        nb_actions: action space size
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(V * F, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, nb_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (batch_size, V, F)
        output: shape (batch_size, nb_actions)
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.net(x)
