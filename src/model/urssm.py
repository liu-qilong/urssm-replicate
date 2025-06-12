import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.fmap import fmap2pointmap
from src.infra.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class URSSM(nn.Module):
    def __init__(self, opt):
        super(URSSM, self).__init__()
        self.opt = opt

        self.feature_extractor = MODEL_REGISTRY[self.opt.network.feature_extractor.name](
            **self.opt.network.feature_extractor.args
        )
        self.fm_solver = MODEL_REGISTRY[self.opt.network.fm_solver.name](
            **self.opt.network.fm_solver.args
        )
        self.permutation = MODEL_REGISTRY[self.opt.network.permutation.name](
            **self.opt.network.permutation.args
        )

    def forward(self, data):
        data_x, data_y = data['first'], data['second']

        # feature extractor
        feat_x = self.feature_extractor(data_x['verts'], data_x['faces'])
        feat_y = self.feature_extractor(data_y['verts'], data_y['faces'])

        # fm solver
        Cxy, Cyx = self.fm_solver(
            feat_x, feat_y,
            data_x['evals'],
            data_y['evals'],
            data_x['evecs_trans'],
            data_y['evecs_trans'],
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

    def inference(self, data):
        data_x, data_y = data['first'], data['second']

        # feature extractor
        feat_x = self.feature_extractor(data_x['verts'], data_x['faces'])
        feat_y = self.feature_extractor(data_y['verts'], data_y['faces'])

        # fm solver
        Cxy, _ = self.fm_solver(
            feat_x, feat_y,
            data_x['evals'],
            data_y['evals'],
            data_x['evecs_trans'],
            data_y['evecs_trans'],
            bidirectional=False,
        )

        # point-wise correspondence
        p2p = []
        Pyx = []
        for it in range(Cxy.shape[0]):
            # todo: make it works on batch
            p2p_ = fmap2pointmap(Cxy[it], data_x['evecs'][it], data_y['evecs'][it])
            C_ = data_y['evecs_trans'][it] @ data_x['evecs'][it][p2p_]
            P_ = data_y['evecs'][it] @ C_ @ data_x['evecs_trans'][it]

            p2p.append(p2p_)
            Pyx.append(P_)

        return {
            'Cxy': Cxy,
            'Pyx': Pyx,
            'p2p': p2p,
        }