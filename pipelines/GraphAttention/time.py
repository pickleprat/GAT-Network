import torch.nn as nn 
import torch 

class TimeOrientedGAT(nn.Module): 
    def __init__(self, transformed_window: int, temporal_patterns: int, negative_slope: float = 0.01): 
        super(TimeOrientedGAT, self).__init__()
        self.transformed_window = transformed_window
        self.temporal_patterns = temporal_patterns
        self.negative_slope = negative_slope
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cat_x  = torch.concat((
            x[:, :, None, :].expand(-1, -1, self.transformed_window, -1), 
            x[:, None, :, :].expand(-1, self.transformed_window, -1, -1)
        ), dim=-1)

        w = nn.Linear(in_features=2*self.temporal_patterns, out_features=1, bias=False)
        leaky_relu = nn.LeakyReLU(negative_slope=self.negative_slope)
        alpha = torch.softmax(leaky_relu(w(cat_x)), dim=2)
        alpha = alpha.squeeze(-1)

        return alpha @ x 