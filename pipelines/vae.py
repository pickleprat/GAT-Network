import torch.nn as nn 
import torch 

class VAutoEncoder(nn.Module): 
    def __init__(
        self, 
        temporal_patterns: int, 
        latent_dim: int, 
        hidden_size: int, 
        decoder_layers: int, 
        channels: int, 
    ): 
        self.temporal_patterns = temporal_patterns 
        self.latent_dim = latent_dim 
        self.decoder_layers = decoder_layers
        self.hidden_size = hidden_size
        self.channels = channels 
        
        self.logvar_layer = nn.Sequential([
            nn.Linear(self.temporal_patterns, self.latent_dim), 
            nn.Tanh(), 
            nn.Linear(self.latent_dim, self.latent_dim), 
            nn.Tanh(), 
            nn.Linear(self.latent_dim, self.latent_dim), 
            nn.Tanh()
        ]) 

        self.mu_layer = nn.Sequential([
            nn.Linear(self.temporal_patterns, self.latent_dim), 
            nn.Tanh(), 
            nn.Linear(self.latent_dim, self.latent_dim), 
            nn.Tanh(), 
            nn.Linear(self.latent_dim, self.latent_dim), 
            nn.Tanh()
        ]) 

        self.decoder = nn.GRU(
            input_size=self.latent_dim, 
            hidden_size=self.hidden_size,  
            num_layers=self.decoder_layers, 
            batch_first=True, 
        )

        self.reconstructor = nn.Linear(
            in_features=self.hidden_size, 
            out_features=self.channels
        )

    def forward(self, x: torch.Tensor): 
        logvariance = self.logvar_layer(x)
        mean = self.mu_layer(x)

        z = mean + torch.exp(0.5 * logvariance) * torch.rand_like(logvariance)
        z = z.expand(-1, 100, -1)

        output, _ = self.decoder(z)
        return self.reconstructor(output)
        
        

        

        