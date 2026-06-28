import torch.nn as nn 
import GraphAttention.feature as fattn 
import GraphAttention.time as tattn

from encoder import Encoder 
from vae import VAutoEncoder

import torch 


class MultiVariateTSGAT(nn.Module): 
    def __init__(
        self, 
        transformed_window, 
        temporal_patterns,
        machine_count: int, 
        features: int, 
        time_window: int, 
        encoder_layers: int, 
        latent_dim: int, 
        negative_slope: int = 0.01,  
        embedding_size: int = 100, 
        kernel_size: int = 7, 
        stride: int = 2, 
        decoder_hidden_size: int = 64, 
        decoder_layers: int = 5
    ): 
    
        self.embedding_size = embedding_size 
        self.transformed_window = transformed_window
        self.temporal_patterns = temporal_patterns
        self.negative_slope  = negative_slope
        self.machine_count = machine_count 
        self.stride = stride 
        self.features = features 
        self.kernel_size = kernel_size
        self.time_window = time_window 
        self.num_layers = encoder_layers 
        self.latent_dim = latent_dim
        self.decoder_layers = decoder_layers
        self.decoder_hidden_size = decoder_hidden_size

        self.transformed_window = (
            self.time_window - self.kernel_size 
        ) // 2 + 1

        self.machine_embedder = nn.Embedding(
            self.machine_count, 
            self.embedding_size, 
        )

        self.conv_layer = nn.Conv1d(
           in_channels=self.features, 
           out_channels=self.temporal_patterns, 
           kernel_size=self.kernel_size, 
           stride=self.stride, 
        )
        
        self.feature_attn = fattn.FeatureOrientedGAT(
            self.transformed_window, 
            self.temporal_patterns, 
            self.negative_slope
        )

        self.time_attn = tattn.TimeOrientedGAT(
            self.transformed_window, 
            self.temporal_patterns, 
            self.negative_slope
        )

        self.encoder = Encoder(
            self.embedding_size, 
            self.temporal_patterns, 
            self.num_layers, 
        )
        
        self.forecaster = nn.Sequential(
            nn.Linear(
                self.temporal_patterns, 
                out_features=self.features
            ), 
            nn.ReLU(), 
            nn.Linear(self.features, self.features), 
            nn.ReLU(), 
            nn.Linear(self.features, self.features), 
            nn.ReLU()
        )

        self.vae = VAutoEncoder(
            temporal_patterns=self.temporal_patterns, 
            latent_dim=self.latent_dim, 
            hidden_size=self.decoder_hidden_size, 
            decoder_layers=self.decoder_layers, 
            channels=self.features
        )
    
    def forward(self, x): 
        machine_ids, sensor, labels = x

        # create machine embeddings 
        machine_embeds = self.machine_embedder(machine_ids)
        trf_x = self.conv_layer(sensor.transpose(1, 2)).transpose(1, 2)

        tattn = self.time_attn(trf_x)
        fattn = self.feature_attn(trf_x)

        # concat features 
        X = torch.concat((
            machine_embeds[:, None, :].expand(-1, self.transformed_window, -1), 
            tattn, 
            trf_x, 
            fattn, 
        ), dim=-1)

        # last hidden state of the encoder for all layers
        hn = self.encoder(X)
        hn = hn[:, -1, :]
        hn = hn[:, None, :]

        forecast = self.forecaster(hn)
        recon = self.vae(hn)

        return forecast, recon

        
        

    

        

        

        

        
            

        
