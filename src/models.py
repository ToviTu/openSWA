import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),  # output: 16 x 1000
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),  # output: 32 x 500
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # output: 64 x 250
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),  # output: 128 x 125
            nn.ReLU(),
        )
        self.encoded_space_dim = 128 * 125

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                16, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            # No activation here, as we're outputting raw signal values
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Adding channel dimension
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.squeeze(1)  # Remove channel dimension for output


class ClassDecoder(nn.Module):
    def __init__(self, encoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.ReLU(),
            nn.Linear(self.encoder.encoded_space_dim, 3),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        encoded = self.encoder.encoder(x)
        return self.net(encoded)
