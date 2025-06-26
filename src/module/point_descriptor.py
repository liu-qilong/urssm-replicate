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


@MODULE_REGISTRY.register()
class WKS(nn.Module):
    # p.s. this module is computed in the whole batch
    # however, computation efficiency doesn't improve much
    def wks(self, evals, evecs, energy_list, sigma, scaled=True):
        # evals: [B, K], evecs: [B, V, K], energy_list: [B, n_descr], sigma: [B]
        coefs = torch.exp(
            - (energy_list[..., None] - torch.log(torch.abs(evals).clamp_min(1e-12))[..., None, :]) ** 2
            / (2 * sigma[..., None, None] ** 2)
        )  # [B, n_descr, K]

        weighted_evecs = evecs.unsqueeze(1) * coefs.unsqueeze(2)  # [B, n_descr, V, K]
        wks = torch.einsum('bnvk,bvk->bnv', weighted_evecs, evecs)  # [B, n_descr, V]

        if scaled:
            inv_scaling = coefs.sum(-1)  # [B, n_descr]
            return wks / (inv_scaling[..., None] + 1e-12)
        else:
            return wks

    def auto_wks(self, evals, evecs, n_descr, scaled=True):
        # evals: [B, K], evecs: [B, V, K]
        abs_ev, _ = evals.abs().sort(dim=1)
        e_min = torch.log(abs_ev[:, 1])
        e_max = torch.log(abs_ev[:, -1])
        sigma = 7 * (e_max - e_min) / n_descr

        e_min = e_min + 2 * sigma
        e_max = e_max - 2 * sigma

        steps = torch.linspace(0, 1, n_descr, device=e_min.device, dtype=e_min.dtype)
        energy_list = e_min[:, None] * (1 - steps) + e_max[:, None] * steps  # [B, n_descr]

        return self.wks(abs_ev, evecs, energy_list, sigma, scaled=scaled)

    def forward(self, evals, evecs, mass, n_descr=128, subsample_step=1, n_eig=200):
        # Truncate eigenpairs
        evals = evals[:, :n_eig]      # [B, K]
        evecs = evecs[:, :, :n_eig]   # [B, V, K]

        feat = self.auto_wks(evals, evecs, n_descr, scaled=True)  # [B, n_descr, V]
        feat = feat[:, :, ::subsample_step]   # [B, n_descr, V_subsampled]
        mass_sub = mass[:, ::subsample_step]  # [B, V_subsampled]

        # Normalize
        # [B, n_descr, V_subsampled], [B, V_subsampled]
        feat_norm = torch.einsum('bnv,bv->bn', feat ** 2, mass_sub)  # [B, n_descr]
        feat = feat / torch.sqrt(feat_norm.unsqueeze(-1) + 1e-12)

        return feat.transpose(1, 2)