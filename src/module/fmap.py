# This implementation is adapted from Dongliang Cao, et al. (2024): https://github.com/dongliangcao/unsupervised-learning-of-robust-spectral-shape-matching. New modules & refinements are added

import torch
import torch.nn as nn

from src.infra.registry import MODULE_REGISTRY


@MODULE_REGISTRY.register()
class RegularizedFMNet(nn.Module):
    """Compute the functional map matrix representation in DPFM"""
    def __init__(self, lmbda=100, resolvant_gamma=0.5):
        super(RegularizedFMNet, self).__init__()
        self.lmbda = lmbda
        self.resolvant_gamma = resolvant_gamma

    def get_mask(self, evals1, evals2, resolvant_gamma):
        def _get_mask(evals1, evals2, resolvant_gamma):
            scaling_factor = max(torch.max(evals1), torch.max(evals2))
            evals1, evals2 = evals1 / scaling_factor, evals2 / scaling_factor
            evals_gamma1 = (evals1 ** resolvant_gamma)[None, :]
            evals_gamma2 = (evals2 ** resolvant_gamma)[:, None]

            M_re = evals_gamma2 / (evals_gamma2.square() + 1) - evals_gamma1 / (evals_gamma1.square() + 1)
            M_im = 1 / (evals_gamma2.square() + 1) - 1 / (evals_gamma1.square() + 1)
            return M_re.square() + M_im.square()
        
        masks = []
        for bs in range(evals1.shape[0]):
            masks.append(_get_mask(evals1[bs], evals2[bs], resolvant_gamma))
        return torch.stack(masks, dim=0)

    def compute_functional_map(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        A = torch.bmm(evecs_trans_x, feat_x)  # [B, K, C]
        B = torch.bmm(evecs_trans_y, feat_y)  # [B, K, C]

        D = self.get_mask(evals_x, evals_y, self.resolvant_gamma)  # [B, K, K]

        A_t = A.transpose(1, 2)  # [B, C, K]
        A_A_t = torch.bmm(A, A_t)  # [B, K, K]
        B_A_t = torch.bmm(B, A_t)  # [B, K, K]

        # row-wise solution
        C_i = []
        for i in range(evals_x.shape[1]):
            D_i = torch.cat(
                [torch.diag(D[bs, i, :].flatten()).unsqueeze(0)
                 for bs in range(evals_x.shape[0])],
                dim=0,
            )
            C = torch.bmm(
                torch.inverse(A_A_t + self.lmbda * D_i),
                B_A_t[:, [i], :].transpose(1, 2),
            )
            C_i.append(C.transpose(1, 2))

        Cxy = torch.cat(C_i, dim=1)
        return Cxy

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, bidirectional=False):
        """
        Forward pass to compute functional map
        Args:
            feat_x (torch.Tensor): feature vector of shape x. [B, Vx, C].
            feat_y (torch.Tensor): feature vector of shape y. [B, Vy, C].
            evals_x (torch.Tensor): eigenvalues of shape x. [B, K].
            evals_y (torch.Tensor): eigenvalues of shape y. [B, K].
            evecs_trans_x (torch.Tensor): pseudo inverse of eigenvectors of shape x. [B, K, Vx].
            evecs_trans_y (torch.Tensor): pseudo inverse of eigenvectors of shape y. [B, K, Vy].

        Returns:
            C (torch.Tensor): functional map from shape x to shape y. [B, K, K].
        """
        Cxy = self.compute_functional_map(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)

        if bidirectional:
            Cyx = self.compute_functional_map(feat_y, feat_x, evals_y, evals_x, evecs_trans_y, evecs_trans_x)
        else:
            Cyx = None

        return Cxy, Cyx


@MODULE_REGISTRY.register()
class RegularizedFMNet_vectorized(nn.Module):
    """Compute the functional map matrix representation in DPFM"""
    def __init__(self, lmbda=100, resolvant_gamma=0.5):
        super(RegularizedFMNet_vectorized, self).__init__()
        self.lmbda = lmbda
        self.resolvant_gamma = resolvant_gamma

    def get_resolvent_mask(self, evals_x, evals_y, resolvant_gamma):
        # evals_x, evals_y: [B, K]
        scaling_factor = torch.max(
            torch.max(evals_x, dim=1, keepdim=True)[0],
            torch.max(evals_y, dim=1, keepdim=True)[0],
        )  # [B, 1]
        evals_x = evals_x / scaling_factor  # [B, K]
        evals_y = evals_y / scaling_factor  # [B, K]
        evals_gamma_x = evals_x ** resolvant_gamma  # [B, K]
        evals_gamma_y = evals_y ** resolvant_gamma  # [B, K]

        # broadcast shapes: [B, K, 1] and [B, 1, K]
        evals_gamma_x = evals_gamma_x.unsqueeze(1)  # [B, 1, K]
        evals_gamma_y = evals_gamma_y.unsqueeze(2)  # [B, K, 1]

        M_re = evals_gamma_y / (evals_gamma_y.square() + 1) - evals_gamma_x / (evals_gamma_x.square() + 1)  # [B, K, K]
        M_im = 1 / (evals_gamma_y.square() + 1) - 1 / (evals_gamma_x.square() + 1)  # [B, K, K]
        return M_re.square() + M_im.square()  # [B, K, K]

    def compute_functional_map(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        A = torch.bmm(evecs_trans_x, feat_x)  # [B, K, C]
        B = torch.bmm(evecs_trans_y, feat_y)  # [B, K, C]

        D = self.get_resolvent_mask(evals_x, evals_y, self.resolvant_gamma)  # [B, K, K]

        A_t = A.transpose(1, 2)  # [B, C, K]
        A_A_t = torch.bmm(A, A_t)  # [B, K, K]
        B_A_t = torch.bmm(B, A_t)  # [B, R, K] p.s. K = R (number of rows of C)
        D_diag = torch.diag_embed(D)  # [B, R, K, K] for each batch and each i, D_diag[b, i] = diag(D[b, i, :])

        # row-wise solution of C_r @ lhs = rhs
        # left-hand side: A_A_t + lambda * D_diag
        A_A_t_expanded = A_A_t.unsqueeze(1).expand(-1, D.shape[1], -1, -1)  # [B, R, K, K]
        lhs = A_A_t_expanded + self.lmbda * D_diag  # [B, R, K, K]

        # right-hand side: B_A_t (for each i, pick [B, i, :])
        rhs = B_A_t.unsqueeze(-2) # [B, R, 1, K]

        # solve batched linear system
        C = torch.linalg.solve(lhs, rhs, left=False)  # [B, R, 1, K]
        C = C.squeeze(-2)  # [B, R, K]

        return C

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, bidirectional=False):
        """
        Forward pass to compute functional map
        Args:
            feat_x (torch.Tensor): feature vector of shape x. [B, Vx, C].
            feat_y (torch.Tensor): feature vector of shape y. [B, Vy, C].
            evals_x (torch.Tensor): eigenvalues of shape x. [B, K].
            evals_y (torch.Tensor): eigenvalues of shape y. [B, K].
            evecs_trans_x (torch.Tensor): pseudo inverse of eigenvectors of shape x. [B, K, Vx].
            evecs_trans_y (torch.Tensor): pseudo inverse of eigenvectors of shape y. [B, K, Vy].

        Returns:
            C (torch.Tensor): functional map from shape x to shape y. [B, K, K].
        """
        Cxy = self.compute_functional_map(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)

        if bidirectional:
            Cyx = self.compute_functional_map(feat_y, feat_x, evals_y, evals_x, evecs_trans_y, evecs_trans_x)
        else:
            Cyx = None

        return Cxy, Cyx