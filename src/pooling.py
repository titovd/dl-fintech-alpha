import torch
import torch.nn as nn

class TemporalAttentionPooling(nn.Module):
    name2activation = {
        "softmax": nn.Softmax(dim=1),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }

    def __init__(self, in_features, activation=None, kernel_size=1, **params):
        """@TODO: Docs."""
        super().__init__()
        self.in_features = in_features
        activation = activation or "softmax"

        self.attention_pooling = nn.Sequential(
            nn.Conv1d(
                in_channels=in_features,
                out_channels=1,
                kernel_size=kernel_size,
                **params
            ),
            TemporalAttentionPooling.name2activation[activation],
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, history_len, feature_size = x.shape

        x = x.view(batch_size, history_len, -1)
        x_a = x.transpose(1, 2)
        x_attn = (self.attention_pooling(x_a) * x_a).transpose(1, 2)
        x_attn = x_attn.sum(1, keepdim=True)

        return x_attn
