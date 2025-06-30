# This implementation is adapted from Dongliang Cao, et al. (2024): https://github.com/dongliangcao/unsupervised-learning-of-robust-spectral-shape-matching. New modules & refinements are added

import numpy as np
import torch


def nn_query(feat_x, feat_y, dim=-2):
    """
    Find correspondences via nearest neighbor query
    Args:
        feat_x: feature vector of shape x. [V1, C].
        feat_y: feature vector of shape y. [V2, C].
        dim: number of dimension
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2].
    """
    dist = torch.cdist(feat_x, feat_y)  # [V1, V2]
    p2p = dist.argmin(dim=dim)
    return p2p


def fmap2pointmap(C12, evecs_x, evecs_y):
    """
    Convert functional map to point-to-point map

    Args:
        C12: functional map (shape x->shape y). Shape [K, K]
        evecs_x: eigenvectors of shape x. Shape [V1, K]
        evecs_y: eigenvectors of shape y. Shape [V2, K]
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2]
    """
    return nn_query(torch.matmul(evecs_x, C12.t()), evecs_y)


def pointmap2fmap(p2p, evecs_x, evecs_y):
    """
    Convert a point-to-point map to functional map

    Args:
        p2p (np.ndarray): point-to-point map (shape x -> shape y). [Vx]
        evecs_x (np.ndarray): eigenvectors of shape x. [Vx, K]
        evecs_y (np.ndarray): eigenvectors of shape y. [Vy, K]
    Returns:
        C21 (np.ndarray): functional map (shape y -> shape x). [K, K]
    """
    C21 = torch.linalg.lstsq(evecs_x, evecs_y[p2p, :]).solution
    return C21


def fmap2pointmap_vectorized(Cxy, evecs_x, evecs_y, verts_mask_x, verts_mask_y):
    """
    Convert functional map to point-to-point map for a  (padded) batch of data in a vectorized manner

    Args:
        Cxy (torch.Tensor): functional map (shape x->shape y). Shape [B, K, K]
        evecs_x (torch.Tensor): eigenvectors of shape x. Shape [B, V_x, K]
        evecs_y (torch.Tensor): eigenvectors of shape y. Shape [B, V_y, K]
        verts_mask_x (torch.Tensor): mask for vertices in shape x. Shape [B, V_x] with valid points as 1 and padded points as 0.
        verts_mask_y (torch.Tensor): mask for vertices in shape y. Shape [B, V_y] with valid points as 1 and padded points as 0.

    Returns:
        p2p (torch.Tensor): point-to-point map (shape y -> shape x). Shape [B, V_y]
    """
    # compute point-to-point map from y to x using the fmap Cxy
    dist = torch.cdist(
        evecs_y,  # Phi_y [B, V_y, K]
        torch.bmm(evecs_x, Cxy.transpose(1, 2)),  # Phi_x @ C_xy^T [B, V_x, K]
    )  # [B, V_y, V_x]

    # mask out padded points:
    inf = torch.finfo(dist.dtype).max  # use large finite value to avoid nan propagation
    dist = dist + (1 - verts_mask_y.float().unsqueeze(-1)) * inf
    dist = dist + (1 - verts_mask_x.float().unsqueeze(-2)) * inf

    # compute point-to-point map from y to x
    p2p = torch.argmin(dist, dim=-1) # [B, V_y]

    # set indices for masked y to -1
    p2p = torch.where(verts_mask_y.bool(), p2p, -1)
    
    return p2p.long()


def p2p_to_permutation_mat_sparse(p2p, V_x):
    """
    Convert point-to-point mapping to a sparse permutation matrix.

    Args:
        p2p (torch.Tensor): Point-to-point mapping of shape [B, V_y], where V_y is the number of vertices in the second mesh.
        V_x (int): Number of vertices in the first mesh (including the padded points).

    Returns:
        torch.sparse_coo_tensor: Sparse permutation matrix of shape [B, V_y, V_x].
    """
    B, V_y = p2p.shape
    device = p2p.device

    valid = (p2p != -1) # [B, V_y]
    row_idx = torch.arange(V_y, device=device).view(1, -1).expand(B, -1) # [B, V_y]
    batch_idx = torch.arange(B, device=device).view(-1, 1).expand(-1, V_y) # [B, V_y]
    col_idx = p2p

    indices = torch.stack([
        batch_idx[valid],
        row_idx[valid],
        col_idx[valid],
    ], dim=0)  # [3, N]
    values = torch.ones(indices.shape[1], dtype=torch.float32, device=device)
    shape = (B, V_y, V_x)

    return torch.sparse_coo_tensor(
        indices,
        values,
        shape,
        device=device,
    )


def pointmap2fmap_vectorized(p2p, evecs_x, evecs_trans_y):
    """
    Convert point-to-point mapping to functional map for a  (padded) batch of data in a vectorized manner.

    Args:
        p2p (torch.Tensor): Point-to-point mapping of shape [B, V_y], where V_y is the number of vertices in the second mesh.
        evecs_x (torch.Tensor): Eigenvectors of shape x. Shape [B, V_x, K].
        evecs_trans_y (torch.Tensor): Transposed eigenvectors of shape y. Shape [B, K, V_y].

    Returns:
        torch.Tensor: Functional map (shape x -> shape y). Shape [B, K, K].
    """
    Pyx = p2p_to_permutation_mat_sparse(p2p, evecs_x.shape[1]) # [V_y, V_x]
    return torch.bmm(
        evecs_trans_y, # [B, K, V_y]
        torch.bmm(Pyx, evecs_x), # [B, V_y, V_x] @ [B, V_y, K] -> [B, V_y, K]
    ) # [B, K, V_x] -> [B, K, K]