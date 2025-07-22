"""Implementation of some naive baselines."""
import torch
from torch import nn

from src.utils.fmap import fmap2pointmap_vectorized, pointmap2Pyx_smooth_vectorized, pointmap2Cxy_vectorized
from src.infra.registry import NETWORK_REGISTRY, MODULE_REGISTRY

@NETWORK_REGISTRY.register()
class HandcraftFMap(nn.Module):
    """
    Functional maps network that only uses handcrafted point descriptors.
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.dummy_params = nn.Parameter(torch.zeros(1))

        self.point_descriptor = MODULE_REGISTRY[self.opt.network.point_descriptor.name](
            **self.opt.network.point_descriptor.args
        )
        self.fm_solver = MODULE_REGISTRY[self.opt.network.fm_solver.name](
            **self.opt.network.fm_solver.args
        )
        self.permute_mat = MODULE_REGISTRY[self.opt.network.permute_mat.name](
            **self.opt.network.permute_mat.args
        )

    def forward(self, data):
        # point descriptor
        des_x = self.point_descriptor(
            evals=data['first']['evals'],
            evecs=data['first']['evecs'],
            mass=data['first']['mass'],
        )
        des_y = self.point_descriptor(
            evals=data['second']['evals'],
            evecs=data['second']['evecs'],
            mass=data['second']['mass'],
        )

        # fm solver
        Cxy, Cyx = self.fm_solver(
            feat_x=des_x,
            feat_y=des_y,
            evals_x=data['first']['evals'],
            evals_y=data['second']['evals'],
            evecs_trans_x=data['first']['evecs_trans'],
            evecs_trans_y=data['second']['evecs_trans'],
            bidirectional=True,
        )

        # point-wise correspondence
        Pxy = self.permute_mat(
            feat_x=des_x,
            feat_y=des_y,
            verts_mask_x=data['first']['verts_mask'],
            verts_mask_y=data['second']['verts_mask'],
        )
        Pyx = self.permute_mat(
            feat_x=des_y,
            feat_y=des_x,
            verts_mask_x=data['second']['verts_mask'],
            verts_mask_y=data['first']['verts_mask'],
        )

        return {
            'Cxy': Cxy,
            'Cyx': Cyx,
            'Pxy': Pxy,
            'Pyx': Pyx,
        }


@NETWORK_REGISTRY.register()
class IdentityFMap(nn.Module):
    """
    Functional maps network that always returns identity maps.
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.dummy_params = nn.Parameter(torch.zeros(1))

    def forward(self, data):
        current_device = next(self.parameters()).device

        # fmap
        B, K = data['first']['evals'].shape
        Cxy = torch.eye(K).repeat(B, 1, 1).to(current_device)
        Cyx = torch.eye(K).repeat(B, 1, 1).to(current_device)

        # point-wise correspondence
        p2p = fmap2pointmap_vectorized(
            Cxy=Cxy,
            evecs_x=data['first']['evecs'],
            evecs_y=data['second']['evecs'],
            verts_mask_x=data['first']['verts_mask'],
            verts_mask_y=data['second']['verts_mask'],
        )
        Pyx = pointmap2Pyx_smooth_vectorized(
            p2p=p2p,
            evecs_x=data['first']['evecs'],
            evecs_y=data['second']['evecs'],
            evecs_trans_x=data['first']['evecs_trans'],
            evecs_trans_y=data['second']['evecs_trans'],
        )
        Pxy = Pyx.transpose(1, 2)

        return {
            'Cxy': Cxy,
            'Cyx': Cyx,
            'Pxy': Pxy,
            'Pyx': Pyx,
        }


@NETWORK_REGISTRY.register()
class IdentityPermutationMat(nn.Module):
    """
    Functional maps network that always yeilds a identity permutation matrix.
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.dummy_params = nn.Parameter(torch.zeros(1))

    def forward(self, data):
        current_device = next(self.parameters()).device

        # point-wise correspondence
        B, V_x, K = data['first']['evecs'].shape
        V_y = data['second']['evecs'].shape[1]
        Pxy = torch.eye(V_x, V_y).repeat(B, 1, 1).to(current_device) \
            * data['first']['verts_mask'].unsqueeze(-1) \
            * data['second']['verts_mask'].unsqueeze(-2) # [B, V_x, V_y] * [B, V_x, 1] * [B, 1, V_y]
        Pyx = Pxy.transpose(1, 2)

        # fmap
        p2p = torch.arange(V_y).repeat(B, 1).to(current_device) # [B, V_y]
        p2p = torch.where(
            p2p > torch.tensor(data['second']['num_verts']).to(current_device).unsqueeze(-1),
            p2p, -1,
        ) # mask out invalid vertex of shape x
        p2p = torch.where(data['second']['verts_mask'].bool(), p2p, -1)  # mask out invalid vertex of shape y
        Cxy = pointmap2Cxy_vectorized(
            p2p=p2p,
            evecs_x=data['first']['evecs'],
            evecs_trans_y=data['second']['evecs_trans'],
        )
        Cyx = Cxy.transpose(1, 2)

        return {
            'Cxy': Cxy,
            'Cyx': Cyx,
            'Pxy': Pxy,
            'Pyx': Pyx,
        }