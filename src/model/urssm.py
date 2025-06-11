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

        self.feature_extractor = MODEL_REGISTRY[self.opt.model.feature_extractor.name](
            **self.opt.model.feature_extractor.args
        )
        self.fm_solver = MODEL_REGISTRY[self.opt.model.fm_solver.name](
            **self.opt.model.fm_solver.args
        )
        self.permutation = MODEL_REGISTRY[self.opt.model.permutation.name](
            **self.opt.model.permutation.args
        )

    def forward(self, data_x, data_y):
        # feature extractor
        feat_x = self.feature_extractor(data_x['verts'], data_x['faces'])
        feat_y = self.feature_extractor(data_y['verts'], data_y['faces'])

        if self.training == True:
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

        else:
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
            for C in Cxy:
                # todo: make it works on batch
                p2p_ = fmap2pointmap(C, data_x['evecs'], data_y['evecs'])
                C_ = data_y['evecs_trans'] @ data_x['evecs'][p2p_]
                P_ = data_y['evecs'] @ C_ @ data_x['evecs_trans']

                p2p.append(p2p_)
                Pyx.append(P_)

            return {
                'Cxy': Cxy,
                'Pyx': Pyx,
                'p2p': p2p,
            }