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


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.encode = nn.Sequential(
            nn.Conv1d(1, 16, 5),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
        )

        self.label_head = nn.Sequential(
            nn.Linear(64 * 246, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        x = self.encode(x)
        x = x.flatten(start_dim=1)
        x = self.label_head(x)
        return x


class AttenNet(nn.Module):
    def __init__(self, hidden_size, num_classes, classifier, num_layers=4):
        super(AttenNet, self).__init__()

        self.seq_embedding = classifier

        self.upsample = nn.Linear(3, hidden_size)

        self.positional_embedding = nn.Parameter(
            torch.randn(seq_len, hidden_size) / hidden_size**0.5
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4,
            activation=nn.GELU(),
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        logit = []
        for i in range(x.size(1)):
            logit.append(self.seq_embedding(x[:, [i]]))
        x = torch.stack(logit, dim=1)
        # x = F.softmax(x, dim=-1)
        x = self.upsample(x)
        x = x.view(x.size(0), seq_len, -1)
        x = x + self.positional_embedding[:seq_len].unsqueeze(0)
        x = self.transformer(x)
        x = self.classifier(x)
        return x
