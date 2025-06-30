import torch
from torch import nn
import torch.nn.functional as F

from src.utils.fmap import fmap2pointmap
from src.infra.registry import NETWORK_REGISTRY, MODULE_REGISTRY

@NETWORK_REGISTRY.register()
class URSSM(nn.Module):
    def __init__(self, opt):
        super(URSSM, self).__init__()
        self.opt = opt

        self.feature_extractor = MODULE_REGISTRY[self.opt.network.feature_extractor.name](
            **self.opt.network.feature_extractor.args
        )
        self.fm_solver = MODULE_REGISTRY[self.opt.network.fm_solver.name](
            **self.opt.network.fm_solver.args
        )
        self.permutation = MODULE_REGISTRY[self.opt.network.permutation.name](
            **self.opt.network.permutation.args
        )

    def cal(self, verts_x, verts_y, faces_x, faces_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        # feature extractor
        feat_x = self.feature_extractor(verts_x, faces_x)
        feat_y = self.feature_extractor(verts_y, faces_y)

        # fm solver
        Cxy, Cyx = self.fm_solver(
            feat_x, feat_y,
            evals_x, evals_y,
            evecs_trans_x, evecs_trans_y,
            bidirectional=True,
        )

        # point-wise correspondence
        similarity = torch.bmm(
            F.normalize(feat_x, dim=-1, p=2),
            F.normalize(feat_y, dim=-1, p=2).transpose(1, 2),
        )

        Pxy = self.permutation(similarity)
        Pyx = self.permutation(similarity.transpose(1, 2))

        return {
            'Cxy': Cxy,
            'Cyx': Cyx,
            'Pxy': Pxy,
            'Pyx': Pyx,
        }

    def forward(self, data):
        verts_x = data['first']['verts']
        verts_y = data['second']['verts']
        faces_x = data['first']['faces']
        faces_y = data['second']['faces']
        evals_x = data['first']['evals']
        evals_y = data['second']['evals']
        evecs_trans_x = data['first']['evecs_trans']
        evecs_trans_y = data['second']['evecs_trans']

        return self.cal(verts_x, verts_y, faces_x, faces_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)

    def inference(self, verts_x, verts_y, faces_x, faces_y, evals_x, evals_y, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y):
        # feature extractor
        feat_x = self.feature_extractor(verts_x, faces_x)
        feat_y = self.feature_extractor(verts_y, faces_y)

        # fm solver
        Cxy, _ = self.fm_solver(
            feat_x, feat_y,
            evals_x, evals_y,
            evecs_trans_x, evecs_trans_y,
            bidirectional=False,
        )

        # point-wise correspondence
        p2p = []
        Pyx = []
        for it in range(Cxy.shape[0]):
            # todo: make it works on batch
            p2p_ = fmap2pointmap(Cxy[it], evecs_x[it], evecs_y[it])
            C_ = evecs_trans_y[it] @ evecs_x[it][p2p_]
            P_ = evecs_y[it] @ C_ @ evecs_trans_x[it]

            p2p.append(p2p_)
            Pyx.append(P_)

        return {
            'Cxy': Cxy,
            'Pyx': Pyx,
            'p2p': p2p,
        }


@NETWORK_REGISTRY.register()
class URSSM_vectorized(nn.Module):
    def __init__(self, opt):
        super(URSSM_vectorized, self).__init__()
        self.opt = opt

        self.point_descriptor = MODULE_REGISTRY[self.opt.network.point_descriptor.name](
            **self.opt.network.point_descriptor.args
        )
        self.feature_extractor = MODULE_REGISTRY[self.opt.network.feature_extractor.name](
            **self.opt.network.feature_extractor.args
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

        # feature extractor
        feat_x = self.feature_extractor(
            x_in=des_x,
            verts_mask=data['first']['verts_mask'],
            mass=data['first']['mass'],
            evals=data['first']['evals'],
            evecs=data['first']['evecs'],
            gradX=torch.sparse_coo_tensor(
                data['first']['gradX_indices'],
                data['first']['gradX_values'],
                data['first']['gradX_size'],
            ),
            gradY=torch.sparse_coo_tensor(
                data['first']['gradY_indices'],
                data['first']['gradY_values'],
                data['first']['gradY_size'],
            ),
        )
        feat_y = self.feature_extractor(
            x_in=des_y,
            verts_mask=data['second']['verts_mask'],
            mass=data['second']['mass'],
            evals=data['second']['evals'],
            evecs=data['second']['evecs'],
            gradX=torch.sparse_coo_tensor(
                data['second']['gradX_indices'],
                data['second']['gradX_values'],
                data['second']['gradX_size'],
            ),
            gradY=torch.sparse_coo_tensor(
                data['second']['gradY_indices'],
                data['second']['gradY_values'],
                data['second']['gradY_size'],
            ),
        )

        # fm solver
        Cxy, Cyx = self.fm_solver(
            feat_x=feat_x,
            feat_y=feat_y,
            evals_x=data['first']['evals'],
            evals_y=data['second']['evals'],
            evecs_trans_x=data['first']['evecs_trans'],
            evecs_trans_y=data['second']['evecs_trans'],
            bidirectional=True,
        )

        # point-wise correspondence
        Pxy = self.permute_mat(
            feat_x=feat_x,
            feat_y=feat_y,
            verts_mask_x=data['first']['verts_mask'],
            verts_mask_y=data['second']['verts_mask'],
        )
        Pyx = self.permute_mat(
            feat_x=feat_y,
            feat_y=feat_x,
            verts_mask_x=data['second']['verts_mask'],
            verts_mask_y=data['first']['verts_mask'],
        )

        return {
            'Cxy': Cxy,
            'Cyx': Cyx,
            'Pxy': Pxy,
            'Pyx': Pyx,
        }