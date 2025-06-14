# This implementation is adapted from Dongliang Cao, et al. (2024): https://github.com/dongliangcao/unsupervised-learning-of-robust-spectral-shape-matching

import torch
import torch.nn as nn

from src.infra.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class SquaredFrobeniusLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        loss = torch.sum(torch.abs(a - b) ** 2, dim=(-2, -1))
        return torch.mean(loss)


@LOSS_REGISTRY.register()
class SURFMNetLoss(nn.Module):
    """
    Loss as presented in the SURFMNet paper.
    Orthogonality + Bijectivity + Laplacian Commutativity
    """
    def __init__(self, w_bij=1.0, w_orth=1.0, w_lap=1e-3):
        """
        Init SURFMNetLoss

        Args:
            w_bij (float, optional): Bijectivity penalty weight. Default 1e3.
            w_orth (float, optional): Orthogonality penalty weight. Default 1e3.
            w_lap (float, optional): Laplacian commutativity penalty weight. Default 1.0.
        """
        super(SURFMNetLoss, self).__init__()
        assert w_bij >= 0 and w_orth >= 0 and w_lap >= 0
        self.w_bij = w_bij
        self.w_orth = w_orth
        self.w_lap = w_lap
        self.squared_frobenius = SquaredFrobeniusLoss()

    def cal(self, Cxy, Cyx, evals_x, evals_y):
        """
        bijectivity loss + orthogonality loss + Laplacian commutativity loss

        Args:
            Cxy (torch.Tensor): matrix representation of functional map (1->2). Shape: [N, K, K]
            Cyx (torch.Tensor): matrix representation of functional map (2->1). Shape: [N, K, K]
            evals_x (torch.Tensor): eigenvalues of shape 1. Shape [N, K]
            evals_y (torch.Tensor): eigenvalues of shape 2. Shape [N, K]
        """
        eye = torch.eye(Cxy.shape[1], Cxy.shape[2], device=Cxy.device).unsqueeze(0)
        eye_batch = torch.repeat_interleave(eye, repeats=Cxy.shape[0], dim=0)
        loss = 0.0

        # bijectivity penalty
        if self.w_bij > 0:
            loss += ( \
                self.squared_frobenius(torch.bmm(Cxy, Cyx), eye_batch) +
                self.squared_frobenius(torch.bmm(Cyx, Cxy), eye_batch) \
            ) * self.w_bij

        # orthogonality penalty
        if self.w_orth > 0:
            loss += ( \
                self.squared_frobenius(torch.bmm(Cxy.transpose(1, 2), Cxy), eye_batch) \
                + self.squared_frobenius(torch.bmm(Cyx.transpose(1, 2), Cyx), eye_batch) \
            ) * self.w_orth

        # laplacian commutativity penalty
        if self.w_lap > 0:
            loss += ( self.squared_frobenius(
                    torch.einsum('abc,ac->abc', Cxy, evals_x),
                    torch.einsum('ab,abc->abc', evals_y, Cxy),
                ) \
                + self.squared_frobenius(
                    torch.einsum('abc,ac->abc', Cyx, evals_y),
                    torch.einsum('ab,abc->abc', evals_x, Cyx),
                ) \
            ) * self.w_lap

        return loss

    def forward(self, infer, data):
        Cxy = infer['Cxy']
        Cyx = infer['Cyx']
        evals_x = data['first']['evals']
        evals_y = data['second']['evals']

        return self.cal(Cxy, Cyx, evals_x, evals_y)


@LOSS_REGISTRY.register()
class BijectivityLoss(nn.Module):
    def __init__(self):
        super(BijectivityLoss, self).__init__()
        self.squared_frobenius = SquaredFrobeniusLoss()

    def cal(self, Cxy, Cyx):
        return self.squared_frobenius(torch.bmm(Cxy, Cyx), torch.eye(Cxy.shape[-1], device=Cxy.device))

    def forward(self, infer, data):
        Cxy = infer['Cxy']
        Cyx = infer['Cyx']

        return self.cal(Cxy, Cyx) + self.cal(Cyx, Cxy)


@LOSS_REGISTRY.register()
class OrthogonalityLoss(nn.Module):
    def __init__(self):
        super(OrthogonalityLoss, self).__init__()
        self.squared_frobenius = SquaredFrobeniusLoss()

    def cal(self, Cxy, Cyx):
        return self.squared_frobenius(torch.bmm(Cxy.transpose(-2, -1), Cxy), torch.eye(Cxy.shape[-1], device=Cxy.device))

    def forward(self, infer, data):
        Cxy = infer['Cxy']
        Cyx = infer['Cyx']

        return self.cal(Cxy, Cyx) + self.cal(Cyx, Cxy)


@LOSS_REGISTRY.register()
class LaplacianCommutativityLoss(nn.Module):
    def __init__(self):
        super(LaplacianCommutativityLoss, self).__init__()
        self.squared_frobenius = SquaredFrobeniusLoss()

    def cal(self, Cxy, Cyx, evals_x, evals_y):
        return (self.squared_frobenius(
            torch.einsum('abc,ac->abc', Cxy, evals_x),
            torch.einsum('ab,abc->abc', evals_y, Cxy),
        ) + self.squared_frobenius(
            torch.einsum('abc,ac->abc', Cyx, evals_y),
            torch.einsum('ab,abc->abc', evals_x, Cyx),
        ))

    def forward(self, infer, data):
        Cxy = infer['Cxy']
        Cyx = infer['Cyx']
        evals_x = data['first']['evals']
        evals_y = data['second']['evals']

        return self.cal(Cxy, Cyx, evals_x, evals_y)


@LOSS_REGISTRY.register()
class SpatialSpectralAlignmentLoss(nn.Module):
    def __init__(self):
        super(SpatialSpectralAlignmentLoss, self).__init__()
        self.squared_frobenius = SquaredFrobeniusLoss()

    def cal(self, Cxy, Cyx, Pxy, Pyx, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y):
        Cxy_est = torch.bmm(
            evecs_trans_y,
            torch.bmm(Pyx, evecs_x),
        )

        Cyx_est = torch.bmm(
            evecs_trans_x,
            torch.bmm(Pxy, evecs_y),
        )

        return self.squared_frobenius(Cxy, Cxy_est) + self.squared_frobenius(Cyx, Cyx_est)

    def forward(self, infer, data):
        Cxy = infer['Cxy']
        Cyx = infer['Cyx']
        Pxy = infer['Pxy']
        Pyx = infer['Pyx']
        evecs_x = data['first']['evecs']
        evecs_y = data['second']['evecs']
        evecs_trans_x = data['first']['evecs_trans']
        evecs_trans_y = data['second']['evecs_trans']

        return self.cal(Cxy, Cyx, Pxy, Pyx, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y)
    

@LOSS_REGISTRY.register()
class PartialFmapsLoss(nn.Module):
    def __init__(self, w_bij=1.0, w_orth=1.0):
        """
        Init PartialFmapsLoss
        Args:
            w_bij (float, optional): Bijectivity penalty weight. Default 1.0.
            w_orth (float, optional): Orthogonality penalty weight. Default 1.0.
        """
        super(PartialFmapsLoss, self).__init__()
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
