import torch
import torch.nn as nn

from src.infra.registry import MODULE_REGISTRY

@MODULE_REGISTRY.register()
class HKS(nn.Module):
    def forward(self, evals, evecs, count=16):
        """
        Compute heat kernel signature with auto-scale
        Args:
            evals (torch.Tensor): eigenvalues of Laplacian matrix [B, K]
            evecs (torch.Tensor): eigenvecetors of Laplacian matrix [B, V, K]
            count (int, optional): number of hks. Default 16.
        Returns:
            out (torch.Tensor): heat kernel signature [B, V, count]
        """
        scales = torch.logspace(-2.0, 0.0, steps=count, device=evals.device, dtype=evals.dtype)

        power_coefs = torch.exp(-evals.unsqueeze(1) * scales.unsqueeze(-1)).unsqueeze(1) # [B, 1, S, K]
        terms = power_coefs * (evecs * evecs).unsqueeze(2) # [B, V, S, K]

        out = torch.sum(terms, dim=-1) # [B, V, S]

        return out