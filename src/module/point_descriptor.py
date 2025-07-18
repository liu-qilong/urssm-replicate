import random

import numpy as np
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
    def wks(self, evals, evecs, energy_list, sigma, scaled=False):
        assert sigma > 0, f"Sigma should be positive ! Given value : {sigma}"

        indices = (evals > 1e-5)
        evals = evals[indices]
        evecs = evecs[:, indices]

        coefs = torch.exp(-torch.square(energy_list[:, None] - torch.log(torch.abs(evals))[None, :]) / (2 * sigma ** 2))

        weighted_evecs = evecs[None, :, :] * coefs[:, None, :]
        wks = torch.einsum('tnk,nk->nt', weighted_evecs, evecs)

        if scaled:
            inv_scaling = coefs.sum(1)
            return (1 / inv_scaling)[None, :] * wks
        else:
            return wks


    def auto_wks(self, evals, evecs, n_descr, scaled=True):
        abs_ev = torch.sort(evals.abs())[0]
        e_min, e_max = torch.log(abs_ev[1]), torch.log(abs_ev[-1])
        sigma = 7 * (e_max - e_min) / n_descr

        e_min += 2 * sigma
        e_max -= 2 * sigma

        energy_list = torch.linspace(float(e_min), float(e_max), n_descr, device=evals.device, dtype=evals.dtype)

        return self.wks(abs_ev, evecs, energy_list, sigma, scaled=scaled)


    def forward(self, evals, evecs, mass, n_descr=128, subsample_step=1, n_eig=128):
        feats = []
        for b in range(evals.shape[0]):
            feat = self.auto_wks(evals[b, :n_eig], evecs[b, :, :n_eig], n_descr, scaled=True)
            feat = feat[:, torch.arange(0, feat.shape[1], subsample_step)]
            feat_norm = torch.einsum('np,np->p', feat, mass[b].unsqueeze(1) * feat)
            feat /= torch.sqrt(feat_norm)
            feats += [feat]
        feats = torch.stack(feats, dim=0)
        return feats

@MODULE_REGISTRY.register()
class WKS_vectorized(nn.Module):
    def __init__(self, n_descr=128, subsample_step=1, n_eig=200):
        super().__init__()
        self.n_descr = n_descr
        self.subsample_step = subsample_step
        self.n_eig = n_eig

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

    def forward(self, evals, evecs, mass):
        # Truncate eigenpairs
        evals = evals[:, :self.n_eig]      # [B, K]
        evecs = evecs[:, :, :self.n_eig]   # [B, V, K]

        feat = self.auto_wks(evals, evecs, self.n_descr, scaled=True)  # [B, n_descr, V]
        feat = feat[:, :, ::self.subsample_step]   # [B, n_descr, V_subsampled]
        mass_sub = mass[:, ::self.subsample_step]  # [B, V_subsampled]

        # Normalize
        # [B, n_descr, V_subsampled], [B, V_subsampled]
        feat_norm = torch.einsum('bnv,bv->bn', feat ** 2, mass_sub)  # [B, n_descr]
        feat = feat / torch.sqrt(feat_norm.unsqueeze(-1) + 1e-12)

        return feat.transpose(1, 2)


@MODULE_REGISTRY.register()
class XYZ(nn.Module):
    def euler_angles_to_rotation_matrix(self, theta):
        R_x = torch.tensor([[1, 0, 0], [0, torch.cos(theta[0]), -torch.sin(theta[0])], [0, torch.sin(theta[0]), torch.cos(theta[0])]])
        R_y = torch.tensor([[torch.cos(theta[1]), 0, torch.sin(theta[1])], [0, 1, 0], [-torch.sin(theta[1]), 0, torch.cos(theta[1])]])
        R_z = torch.tensor([[torch.cos(theta[2]), -torch.sin(theta[2]), 0], [torch.sin(theta[2]), torch.cos(theta[2]), 0], [0, 0, 1]])

        matrices = [R_x, R_y, R_z]

        R = torch.mm(matrices[2], torch.mm(matrices[1], matrices[0]))
        return R


    def get_random_rotation(self, x, y, z):
        thetas = torch.zeros(3, dtype=torch.float)
        degree_angles = [x, y, z]
        for axis_ind, deg_angle in enumerate(degree_angles):
            rand_deg_angle = random.random() * 2 * deg_angle - deg_angle
            rand_radian_angle = float(rand_deg_angle * np.pi) / 180.0
            thetas[axis_ind] = rand_radian_angle

        return self.euler_angles_to_rotation_matrix(thetas)


    def forward(self, verts, verts_mask, rot_x=0, rot_y=90.0, rot_z=0, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1):
        if self.training:
            # random rotation
            rotation_matrix = self.get_random_rotation(rot_x, rot_y, rot_z).repeat(verts.shape[0], 1, 1).to(verts.device)
            verts = torch.bmm(verts, rotation_matrix.transpose(1, 2))

            # random noise
            noise = std * torch.randn(verts.shape).to(verts.device)
            noise = noise.clamp(-noise_clip, noise_clip)
            verts += noise

            # random scaling
            scales = [scale_min, scale_max]
            scale = scales[0] + torch.rand((3,)) * (scales[1] - scales[0])
            verts = verts * scale.to(verts.device)

        return verts * verts_mask.unsqueeze(-1)  # (B, V, 3) * (B, V, 1) -> (B, V, 3)


@MODULE_REGISTRY.register()
class XYZ_vectorized(nn.Module):
    def euler_angles_to_rotation_matrix(self, theta):
        # theta: shape (3,) — angles in radians
        # Use torch functions to create rotation matrices that support gradients and device
        cx, cy, cz = torch.cos(theta)
        sx, sy, sz = torch.sin(theta)

        R_x = torch.stack([
            torch.tensor([1., 0., 0.], device=theta.device, dtype=theta.dtype),
            torch.tensor([0., cx, -sx], device=theta.device, dtype=theta.dtype),
            torch.tensor([0., sx,  cx], device=theta.device, dtype=theta.dtype)
        ])
        R_y = torch.stack([
            torch.tensor([cy, 0., sy], device=theta.device, dtype=theta.dtype),
            torch.tensor([0., 1., 0.], device=theta.device, dtype=theta.dtype),
            torch.tensor([-sy, 0., cy], device=theta.device, dtype=theta.dtype)
        ])
        R_z = torch.stack([
            torch.tensor([cz, -sz, 0.], device=theta.device, dtype=theta.dtype),
            torch.tensor([sz,  cz, 0.], device=theta.device, dtype=theta.dtype),
            torch.tensor([0.,   0., 1.], device=theta.device, dtype=theta.dtype)
        ])
        # R = Rz @ Ry @ Rx
        R = R_z @ R_y @ R_x
        return R

    def get_random_rotation(self, x, y, z, device=None, dtype=None):
        # each angle is sampled uniformly from [-angle, angle] degrees
        max_angles = torch.tensor([x, y, z], device=device, dtype=dtype)
        max_rads = torch.deg2rad(max_angles)
        rand = torch.rand(3, device=device, dtype=dtype) * 2 - 1  # [-1, 1]
        thetas = rand * max_rads  # [-max_rad, max_rad] for each axis
        return self.euler_angles_to_rotation_matrix(thetas)

    def forward(self, verts, verts_mask, rot_x=0, rot_y=90.0, rot_z=0, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1):
        if self.training:
            # random rotation
            rot = self.get_random_rotation(
                rot_x, rot_y, rot_z, device=verts.device, dtype=verts.dtype
            ).unsqueeze(0).repeat(verts.shape[0], 1, 1)
            verts = torch.bmm(verts, rot.transpose(1, 2))

            # random noise
            noise = std * torch.randn_like(verts)
            noise = noise.clamp(-noise_clip, noise_clip)
            verts += noise

            # random scaling
            scale = scale_min + torch.rand(3, device=verts.device, dtype=verts.dtype) * (scale_max - scale_min)
            verts = verts * scale

        return verts * verts_mask.unsqueeze(-1)  # (B, V, 3) * (B, V, 1) -> (B, V, 3)