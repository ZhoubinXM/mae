import torch.nn as nn
from src.utils.weight_init import weight_init
from src.model.layers.transformer_blocks import RMSNorm


class MultimodalDecoder(nn.Module):
    """A naive MLP-based multimodal decoder"""

    def __init__(self, embed_dim, future_steps, hidden_dim=256) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.future_steps = future_steps

        self.multimodal_proj = nn.Linear(embed_dim, 6 * embed_dim)

        self.loc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, future_steps * 2),
        )
        self.pi = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x):
        x = self.multimodal_proj(x).view(-1, 6, self.embed_dim)
        loc = self.loc(x).view(-1, 6, self.future_steps, 2)
        pi = self.pi(x).squeeze(-1)

        return loc, pi

class MultimodalPiDecoder(nn.Module):
    """A naive MLP-based multimodal decoder"""

    def __init__(self, embed_dim, future_steps, hidden_dim=256,
                 norm_layer=RMSNorm,
                 act_layer=nn.GELU) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.future_steps = future_steps

        # self.multimodal_proj = nn.Linear(embed_dim, 6 * embed_dim)

        self.pi = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(hidden_dim, embed_dim),
            norm_layer(hidden_dim),
            act_layer(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(weight_init)

    def forward(self, x):
        # x = self.multimodal_proj(x).view(-1, 6, self.embed_dim)
        pi = self.pi(x).squeeze(-1)

        return pi

class SEPTMultimodalDecoder(nn.Module):
    """A naive MLP-based multimodal decoder"""

    def __init__(self, embed_dim, future_steps, hidden_dim=512) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.future_steps = future_steps

        self.loc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, future_steps * 2),
        )
        self.pi = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x):
        loc = self.loc(x).view(-1, 6, self.future_steps, 2)
        pi = self.pi(x).squeeze(-1)

        return loc, pi

class SEPTProposeDecoder(nn.Module):
    """A naive MLP-based multimodal decoder"""

    def __init__(self, embed_dim, future_steps, hidden_dim=512) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.future_steps = future_steps

        self.loc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, future_steps * 2),
        )

    def forward(self, x):
        loc = self.loc(x).view(-1, 6, self.future_steps, 2)

        return loc

class SEPTMAEDecoder(nn.Module):
    """A naive MLP-based multimodal decoder"""

    def __init__(self, embed_dim, future_steps, hidden_dim=512) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.future_steps = future_steps

        self.loc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, future_steps * 2),
        )

    def forward(self, x):
        loc = self.loc(x).view(-1, self.future_steps, 2)

        return loc
    

class MultiAgentDecoder(nn.Module):
    """A naive MLP-based multimodal decoder"""

    def __init__(self, embed_dim, future_steps, hidden_dim=512) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.future_steps = future_steps

        self.loc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, future_steps * 2),
        )
        # self.pi = nn.Sequential(
        #     nn.Linear(embed_dim, hidden_dim),
        #     # nn.LayerNorm(hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, embed_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(embed_dim, 1),
        # )

    def forward(self, x):
        B, N, m, _ = x.shape
        loc = self.loc(x).view(B, N, m, self.future_steps, 2)
        # pi = self.pi(x).squeeze(-1).view(B, N, m)

        return loc
    
class MultiAgentProposeDecoder(nn.Module):
    """A naive MLP-based multimodal decoder"""

    def __init__(self, embed_dim, future_steps, hidden_dim=512) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.future_steps = future_steps

        self.loc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, future_steps * 2),
        )

    def forward(self, x):
        B, N, m, _ = x.shape
        loc = self.loc(x).view(B, N, m, self.future_steps, 2)

        return loc