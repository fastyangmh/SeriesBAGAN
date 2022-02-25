#import
import torch.nn as nn


#class
class Encoder(nn.Module):
    def __init__(self, in_features, z_dim) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features * 2),
            nn.BatchNorm1d(num_features=in_features * 2), nn.GLU(dim=-1))
        self.mu_layer = nn.Linear(in_features=in_features, out_features=z_dim)
        self.sigma_layer = nn.Linear(in_features=in_features,
                                     out_features=z_dim)

    def forward(self, x):
        x = self.layers(x)
        mu = self.mu_layer(x)
        sigma = self.sigma_layer(x)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, in_features, z_dim) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=z_dim * 4),
            nn.BatchNorm1d(num_features=z_dim * 4), nn.GLU(dim=-1),
            nn.Linear(in_features=z_dim * 2, out_features=in_features),
            nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)


class Generator(Decoder):
    def __init__(self, in_features, z_dim) -> None:
        super().__init__(in_features, z_dim)


class Discriminator(Encoder):
    def __init__(self, in_features, z_dim, num_classes) -> None:
        super().__init__(in_features, z_dim)
        # real or fake class and classes
        # 1 + num_classes
        self.classifier = nn.Linear(in_features=in_features,
                                    out_features=num_classes + 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layers(x)
        return x, self.sigmoid(self.classifier(x))
