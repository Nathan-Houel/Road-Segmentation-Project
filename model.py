import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    (Conv3x3 -> BN -> ReLU) * 2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()       # Une liste spéciale PyTorch pour stocker nos couches
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # L'outil pour diviser la taille par 2

        # Encodeur
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Décodeur
        self.ups = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        # Couche de Classification Pixel par Pixel
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)


    def forward(self, x):
        skip_connections = [] # Liste locale ici 

        # Descente
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Remontée
        skip_connections = skip_connections[::-1] # On inverse la liste des souvenirs pour que le premier souvenir corresponde au dernier étage (le plus profond)

        # On boucle de 0 à la fin, par pas de 2
        for idx in range(0, len(self.ups), 2):
            # Étape A : On remonte (ConvTranspose2d)
            x = self.ups[idx](x)
            
            # Étape B : On récupère le souvenir correspondant
            skip_connection = skip_connections[idx // 2]
            concat_skip = torch.cat((x, skip_connection), dim=1)
            x = self.ups[idx+1](concat_skip)

        # Sortie
        return self.final_conv(x)