import torch
from torch import nn

from src.model import pos_emb
from src.tool.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class MLP(nn.Module):
    def __init__(self, device, img_size: int = 10):
        super().__init__()
        self.device = device
        self.img_size = img_size
        
        self.position_embedding = nn.Embedding(99, int(self.img_size * self.img_size / 2))
        self.young_embedding = nn.Linear(1, int(self.img_size * self.img_size / 2))

        self.model = nn.Sequential(
            nn.Linear(self.img_size * self.img_size * 2, 256),
            nn.Sigmoid(),
            nn.Linear(256, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 1),
        )

        # remember to send model to device!
        self.to(self.device)

    def forward(self, x):
        img_stack, young = x
        
        # reshape img_stack
        infer_shape = img_stack.shape[:-2] + (self.img_size * self.img_size,)  # e.g. (..., 198, 10, 10) -> (..., 198, 100)
        img_stack = img_stack.reshape(infer_shape)

        # positional embedding
        pos_arr = torch.concat([
            torch.arange(99, device=self.device),
            torch.arange(99, device=self.device),
            ], dim=-1,
        )
        pos_emb = self.position_embedding(pos_arr)
        pos_emb = pos_emb.expand(img_stack.shape[:-1] + (-1,))  # e.g. (198, 50) -> (..., 198, 50)

        # youngs' modulus embedding
        young_emb = self.young_embedding(young.unsqueeze(-1))
        young_emb = young_emb.unsqueeze(-2).expand(img_stack.shape[:-1] + (-1,))  # e.g. (50,) -> (..., 198, 50)

        x = torch.cat([img_stack, pos_emb, young_emb], dim=-1)
        
        return self.model(x).squeeze(-1)
    

@MODEL_REGISTRY.register()
class MLP_cos_emb(nn.Module):
    def __init__(self, device, img_size: int = 10, pos_emb_len: int = 100, young_emb_len: int = 100, hidden: int = 256):
        super().__init__()
        self.device = device
        self.img_size = img_size
        self.pos_emb_len = pos_emb_len
        self.young_emb_len = young_emb_len
        self.hidden = hidden
        
        self.position_embedding = pos_emb.PositionalEncoding(self.device, l=self.pos_emb_len)
        self.young_embedding = pos_emb.PositionalEncoding(self.device, l=self.young_emb_len)

        self.mlp = nn.Sequential(
            nn.Linear(self.img_size * self.img_size + self.pos_emb_len + self.young_emb_len, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden * 4),
            nn.ReLU(),
            nn.Linear(hidden * 4, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # remember to send model to device!
        self.to(self.device)

    def forward(self, x):
        img_stack, young = x
        
        # reshape img_stack
        infer_shape = img_stack.shape[:-2] + (self.img_size * self.img_size,)  # e.g. (..., 198, 10, 10) -> (..., 198, 100)
        img_stack = img_stack.reshape(infer_shape)

        # positional embedding
        pos_arr = torch.arange(198.0, device=self.device)
        pos_emb = self.position_embedding(pos_arr)  # e.g. (198,) -> (198, 50)
        pos_emb = pos_emb.expand(img_stack.shape[:-1] + (-1,))  # e.g. (198, 50) -> (..., 198, 50)

        # youngs' modulus embedding
        young_emb = self.young_embedding(young)  # e.g. (50,) -> (..., 50)
        young_emb = young_emb.unsqueeze(-2).expand(img_stack.shape[:-1] + (-1,))  # e.g. (..., 50) -> (..., 198, 50)

        x = torch.cat([img_stack, pos_emb, young_emb], dim=-1)
        
        return self.mlp(x).squeeze(-1)
    

@MODEL_REGISTRY.register()
class PatchMLP_cos_emb(MLP_cos_emb):
    def forward(self, x):
        img_patch, sensor_id, young = x
        
        # reshape img_stack
        infer_shape = img_patch.shape[:-2] + (self.img_size * self.img_size,)  # e.g. (..., 10, 10) -> (..., 100)
        img_patch = img_patch.reshape(infer_shape)

        # positional embedding
        pos_emb = self.position_embedding(sensor_id)  # e.g. (...,) -> (..., 50)

        # youngs' modulus embedding
        young_emb = self.young_embedding(young)  # (50,) -> (..., 50)

        x = torch.cat([img_patch, pos_emb, young_emb], dim=-1)
        
        return self.mlp(x).squeeze(-1)