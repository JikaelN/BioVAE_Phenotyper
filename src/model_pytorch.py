import torch
from torch import nn, std

class BioVAE(nn.Module):
    def __init__(self):
        super(BioVAE, self).__init__()
        # Encoder: Compresses cell image to vector
        # Bas was simple encoder but to increase accuracy we changed to 3 distincts layer
        self.encoder = nn.Sequential(
            # Layer 1: 28 x 28 to 14 x 14
            nn.Conv2d(3, 32, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(32), # Normalization of input activations
            nn.LeakyReLU(0.2),  # Leaky ReLU prevents "dead neurons"
            
            #layer 2: 14 x 14 to 7 x 7
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # Layer 3: 7 x 7 to 4 x 4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Flatten()
        )

        # Latent space (Mean and Log-Variance)
        self.fc_mu = nn.Linear(128 * 4 * 4, 20)
        self.fc_logvar = nn.Linear(128 * 4 * 4, 20)
        
        # Decoder: Reconstructs cell image from vector
        self.decoder_input = nn.Linear(20, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        x_encoded = self.encoder(x)
        mu = self.fc_mu(x_encoded)
        logvar = self.fc_logvar(x_encoded)
        z = self.reparameterize(mu, logvar)
        z_reshaped = self.decoder_input(z).view(-1, 128, 4, 4)
        x_recon = self.decoder(z_reshaped)
        return x_recon, mu, logvar
        

