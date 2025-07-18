# This implementation is adapted from Dongliang Cao, et al. (2024): https://github.com/dongliangcao/unsupervised-learning-of-robust-spectral-shape-matching. New modules & refinements are added

import torch

from src.loss import BaseLoss
from src.infra.registry import LOSS_REGISTRY

class SquaredFrobeniusLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        # a [B, ...] and b [B, ...] should have the same shape
        batch_num = a.shape[0]
        loss = torch.sum(torch.abs(a - b) ** 2, dim=(-2, -1))
        return torch.mean(loss) * batch_num  # scale by batch size to get the summed rather than the averaged loss over all samples


@LOSS_REGISTRY.register()
class BijectivityLoss(BaseLoss):
    def __init__(self):
        super().__init__()
        self.squared_frobenius = SquaredFrobeniusLoss()

    def forward(self, Cxy, Cyx):
        return self.squared_frobenius(
            torch.bmm(Cxy, Cyx),
            torch.eye(Cxy.shape[-1], device=Cxy.device),
        )

    def feed(self, infer, data):
        Cxy = infer['Cxy']
        Cyx = infer['Cyx']
        loss_val = self(Cxy, Cyx) + self(Cyx, Cxy)
        
        # if self.eval() is called, gather total loss
        if not self.training:
            self.loss_total += loss_val
            self.sample_total += len(Cxy)
        
        return loss_val


@LOSS_REGISTRY.register()
class OrthogonalityLoss(BaseLoss):
    def __init__(self):
        super().__init__()
        self.squared_frobenius = SquaredFrobeniusLoss()

    def forward(self, Cxy):
        return self.squared_frobenius(
            torch.bmm(Cxy.transpose(-2, -1), Cxy),
            torch.eye(Cxy.shape[-1], device=Cxy.device),
        )

    def feed(self, infer, data):
        Cxy = infer['Cxy']
        Cyx = infer['Cyx']
        loss_val = self(Cxy) + self(Cyx)

        # if self.eval() is called, gather total loss
        if not self.training:
            self.loss_total += loss_val
            self.sample_total += len(Cxy)
        
        return loss_val


@LOSS_REGISTRY.register()
class LaplacianCommutativityLoss(BaseLoss):
    def __init__(self):
        super().__init__()
        self.squared_frobenius = SquaredFrobeniusLoss()

    def forward(self, Cxy, Cyx, evals_x, evals_y):
        return (self.squared_frobenius(
            torch.einsum('abc,ac->abc', Cxy, evals_x),
            torch.einsum('ab,abc->abc', evals_y, Cxy),
        ) + self.squared_frobenius(
            torch.einsum('abc,ac->abc', Cyx, evals_y),
            torch.einsum('ab,abc->abc', evals_x, Cyx),
        ))

    def feed(self, infer, data):
        Cxy = infer['Cxy']
        Cyx = infer['Cyx']
        evals_x = data['first']['evals']
        evals_y = data['second']['evals']
        loss_val = self(Cxy, Cyx, evals_x, evals_y)
        
        # if self.eval() is called, gather total loss
        if not self.training:
            self.loss_total += loss_val
            self.sample_total += len(Cxy)
        
        return loss_val


@LOSS_REGISTRY.register()
class SpatialSpectralAlignmentLoss(BaseLoss):
    def __init__(self):
        super().__init__()
        self.squared_frobenius = SquaredFrobeniusLoss()

    def forward(self, Cxy, Cyx, Pxy, Pyx, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y):
        Cxy_est = torch.bmm(
            evecs_trans_y,
            torch.bmm(Pyx, evecs_x),
        )

        Cyx_est = torch.bmm(
            evecs_trans_x,
            torch.bmm(Pxy, evecs_y),
        )

        return (self.squared_frobenius(Cxy, Cxy_est) + self.squared_frobenius(Cyx, Cyx_est))

    def feed(self, infer, data):
        Cxy = infer['Cxy']
        Cyx = infer['Cyx']
        Pxy = infer['Pxy']
        Pyx = infer['Pyx']
        evecs_x = data['first']['evecs']
        evecs_y = data['second']['evecs']
        evecs_trans_x = data['first']['evecs_trans']
        evecs_trans_y = data['second']['evecs_trans']
        loss_val = self(Cxy, Cyx, Pxy, Pyx, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y)
        
        # if self.eval() is called, gather total loss
        if not self.training:
            self.loss_total += loss_val
            self.sample_total += len(Cxy)
        
        return loss_val
    

@LOSS_REGISTRY.register()
class PartialFmapsLoss(BaseLoss):
    def __init__(self, w_bij=1.0, w_orth=1.0):
        """
        Init PartialFmapsLoss
        Args:
            w_bij (float, optional): Bijectivity penalty weight. Default 1.0.
            w_orth (float, optional): Orthogonality penalty weight. Default 1.0.
        """
        super().__init__()
        assert w_bij >= 0 and w_orth >= 0, 'Loss weight should be non-negative.'
        self.w_bij = w_bij
        self.w_orth = w_orth
        self.squared_frobenius = SquaredFrobeniusLoss()

    def forward(self, C_fp, C_pf, evals_full, evals_partial):
        assert C_fp.shape[0] == 1, 'Currently, only support batch size = 1'
        C_fp, C_pf = C_fp[0], C_pf[0]
        evals_full, evals_partial = evals_full[0], evals_partial[0]

        # compute area ratio between full shape and partial shape r
        r = min((evals_partial < evals_full.max()).sum(), C_fp.shape[0] - 1)
        eye = torch.zeros_like(C_fp)
        eye[torch.arange(0, r + 1), torch.arange(0, r + 1)] = 1.0

        if self.w_bij > 0:
            bijectivity_loss = self.w_bij * self.squared_frobenius(torch.matmul(C_fp, C_pf), eye)
        else:
            bijectivity_loss = 0.0

        if self.w_orth > 0:
            orthogonality_loss = self.w_bij * self.squared_frobenius(torch.matmul(C_fp, C_fp.t()), eye)
        else:
            orthogonality_loss = 0.0

        return {'l_bij': bijectivity_loss, 'l_orth': orthogonality_loss}
