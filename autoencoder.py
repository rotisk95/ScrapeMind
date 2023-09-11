import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=768, encoding_dim=2048):  # Changed encoding_dim to match SpikingTransformerNetwork's input_dim
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),  # Adjusted intermediate layer
            nn.ReLU(True),
            nn.Linear(1024, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 1024),  # Adjusted intermediate layer
            nn.ReLU(True),
            nn.Linear(1024, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)
