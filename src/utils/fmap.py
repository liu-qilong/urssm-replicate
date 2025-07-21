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


def corr2pointmap_vectorized(corr_x, corr_y, num_verts_y):
    """
    Convert a pair of correspondences to point-to-point map in a vectorized manner.
    Args:
        corr_x (torch.Tensor): Correspondences from template to target. Shape [B, V_t] _P.S. V_t is the number of vertices in the template shape._
        corr_y (torch.Tensor): Correspondences from target to template. Shape [B, V_t]
        num_verts_y (int): Number of vertices in the target shape.
    Returns:
        p2p (torch.Tensor): Point-to-point map (shape y -> shape x). Shape [B, V_y] _P.S. Padded points will have value -1.
    """
    # template -(corr_x)-> shape x <--> shape y <-(corr_y)- target
    # i.e. the i-th row of corr_y is correspongding with the i-th row of corr_x
    B, V_t = corr_x.shape
    batch_idx = torch.arange(B, device=corr_y.device).unsqueeze(1).expand(B, V_t)  # [B, V_t]
    p2p_t = torch.full((B, V_t), -1, dtype=torch.long).to(device=corr_y.device)
    p2p_t[batch_idx, corr_y] = corr_x

    # get p2p in shape [B, V_y]
    V_y = num_verts_y

    if V_t > V_y:
        p2p = p2p_t[:, :V_y]

    else:
        p2p = torch.full((B, V_y), -1, dtype=torch.long).to(device=corr_y.device)
        p2p[:, :V_t] = p2p_t
    
    return p2p


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
        p2p (torch.Tensor): point-to-point map (shape y -> shape x). Shape [B, V_y]. _P.S. Padded points will have value -1._
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

def pointmap2Cxy_vectorized(p2p, evecs_x, evecs_trans_y, verts_mask_y):
    """
    Convert point-to-point mapping to functional map for a  (padded) batch of data in a vectorized manner.

    Args:
        p2p (torch.Tensor): Point-to-point mapping of shape y -> x. Shape [B, V_y].
        evecs_x (torch.Tensor): Eigenvectors of shape x. Shape [B, V_x, K].
        evecs_trans_y (torch.Tensor): Transposed eigenvectors of shape y. Shape [B, K, V_y].
        verts_mask_y (torch.Tensor): mask for vertices in shape y. Shape [B, V_y] with valid points as 1 and padded points as 0.

    Returns:
        torch.Tensor: Functional map (shape x -> shape y). Shape [B, K, K].
    """
    # expand p2p to [B, V, K] & gather along dim=1 (the V dimension)
    # this will get Pyx @ Phi_x in effect
    K = evecs_x.shape[-1]
    index = torch.clamp(p2p, min=0).unsqueeze(-1).expand(-1, -1, K) # [B, V, K]
    permuted_evecs_x = torch.gather(evecs_x, dim=1, index=index) * verts_mask_y.unsqueeze(-1) # permute & mask out invalid rows

    # Cxy = Phi_y^T @ Pyx @ Phi_x
    return evecs_trans_y @ permuted_evecs_x


def pointmap2Pyx_smooth_vectorized(p2p, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y, verts_mask_y):
    """
    Convert point-to-point mapping to the permutation matrix, with spectral smoothing applied.

    Args:
        p2p (torch.Tensor): Point-to-point mapping of shape y -> x. Shape [B, V_y].
        evecs_x (torch.Tensor): Eigenvectors of shape x. Shape [B, V_x, K].
        evecs_y (torch.Tensor): Eigenvectors of shape y. Shape [B, V_y, K].
        evecs_trans_x (torch.Tensor): Transposed eigenvectors of shape x. Shape [B, K, V_x].
        evecs_trans_y (torch.Tensor): Transposed eigenvectors of shape y. Shape [B, K, V_y].
        verts_mask_y (torch.Tensor): mask for vertices in shape y. Shape [B, V_y] with valid points as 1 and padded points as 0.

    Returns:
        torch.tensor: permutation matrix of shape [B, V_y, V_x].
    """
    Cxy = pointmap2Cxy_vectorized(p2p, evecs_x, evecs_trans_y, verts_mask_y)  
    Pyx = evecs_y @ Cxy @ evecs_trans_x

    return Pyx


def pointmap2Pyx_vectorized(p2p, num_verts_y, num_verts_x):
    """
    Convert point-to-point mapping to the permutation matrix in a vectorized manner.

    Args:
        p2p (torch.Tensor): Point-to-point mapping of shape y -> x. Shape [B, V_y].
        num_verts_y (int): Number of vertices in shape y.
        num_verts_x (int): Number of vertices in shape x.
    Returns:
        torch.Tensor: Permutation matrix of shape [B, V_y, V_x].
    """
    B, V_y, V_x = len(p2p), max(num_verts_y), max(num_verts_x)

    # batch indices (broadcasting)
    batch_idx = torch.arange(B).unsqueeze(1).expand(-1, V_y).to(p2p.device)  # [B, V_y]
    v_y_idx = torch.arange(V_y).unsqueeze(0).expand(B, -1).to(p2p.device)  # [B, V_y]
    v_x_idx = torch.clamp(p2p, min=0).to(p2p.device)  # [B, V_y] P.S. clamp -1 as 0 for valid indices

    # set ones where valid
    mask = (p2p != -1)
    Pyx = torch.zeros((B, V_y, V_x), dtype=torch.float32, device=p2p.device)
    Pyx[batch_idx[mask], v_y_idx[mask], v_x_idx[mask]] = 1.0

    return Pyx