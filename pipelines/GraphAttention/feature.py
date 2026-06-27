import torch.nn as nn 
import torch 

class FeatureOrientedGAT(nn.Module): 
    def __init__(self, transformed_window: int, temporal_patterns: int, negative_slope: float = 0.01): 
        super(FeatureOrientedGAT, self).__init__()
        self.transformed_window = transformed_window
        self.temporal_patterns = temporal_patterns
        self.negative_slope = negative_slope
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_x = x.transpose(1, 2)
        cat_x  = torch.concat((
            feature_x[:, :, None, :].expand(-1, -1, self.temporal_patterns, -1), 
            feature_x[:, None, :, :].expand(-1, self.temporal_patterns, -1, -1)
        ), dim=-1)

        w = nn.Linear(in_features=2*self.transformed_window, out_features=1, bias=False)
        leaky_relu = nn.LeakyReLU(negative_slope=self.negative_slope)
        alpha = torch.softmax(leaky_relu(w(cat_x)), dim=2)
        alpha = alpha.squeeze(-1)

        return (alpha @ feature_x).transpose(1, 2)
