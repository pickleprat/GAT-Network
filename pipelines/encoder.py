
import torch.nn as nn 
import torch 

class Encoder(nn.Module): 
    def __init__(self, embedding_size: int, temporal_patterns: int, num_layers: int): 
        super(Encoder, self).__init__()
        self.gru = nn.GRU(
            input_size=3*temporal_patterns + embedding_size, 
            hidden_size=temporal_patterns, 
            num_layers=num_layers, 
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hn = self.gru(x)
        return hn  