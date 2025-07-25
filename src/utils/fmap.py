# This implementation is adapted from Dongliang Cao, et al. (2024): https://github.com/dongliangcao/unsupervised-learning-of-robust-spectral-shape-matching. New modules & refinements are added

import torch
from src.module.fmap import RegularizedFMNet_vectorized


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
        p2p (torch.Tensor): point-to-point map (shape y -> shape x). Shape [B, V_y]. _P.S. Invalid points will have value -1._
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

def pointmap2Cxy_vectorized(p2p, evecs_x, evecs_trans_y):
    """
    Convert point-to-point mapping to functional map for a  (padded) batch of data in a vectorized manner.

    Args:
        p2p (torch.Tensor): Point-to-point mapping of shape y -> x. Shape [B, V_y]. _P.S. Invalid points will have value -1._
        evecs_x (torch.Tensor): Eigenvectors of shape x. Shape [B, V_x, K].
        evecs_trans_y (torch.Tensor): Transposed eigenvectors of shape y. Shape [B, K, V_y].

    Returns:
        torch.Tensor: Functional map (shape x -> shape y). Shape [B, K, K].
    """
    # expand p2p to [B, V, K] & gather along dim=1 (the V dimension)
    # this will get Pyx @ Phi_x in effect
    K = evecs_x.shape[-1]
    mask = (p2p != -1).unsqueeze(-1) # [B, V_y, 1]
    index = torch.clamp(p2p, min=0).unsqueeze(-1).expand(-1, -1, K) # [B, V, K]
    permuted_evecs_x = torch.gather(evecs_x, dim=1, index=index) * mask # permute & mask out invalid points

    # Cxy = Phi_y^T @ Pyx @ Phi_x
    return evecs_trans_y @ permuted_evecs_x


def pointmap2Pyx_smooth_vectorized(p2p, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y):
    """
    Convert point-to-point mapping to the permutation matrix, with spectral smoothing applied.

    Args:
        p2p (torch.Tensor): Point-to-point mapping of shape y -> x. Shape [B, V_y]. _P.S. Invalid points will have value -1._
        evecs_x (torch.Tensor): Eigenvectors of shape x. Shape [B, V_x, K].
        evecs_y (torch.Tensor): Eigenvectors of shape y. Shape [B, V_y, K].
        evecs_trans_x (torch.Tensor): Transposed eigenvectors of shape x. Shape [B, K, V_x].
        evecs_trans_y (torch.Tensor): Transposed eigenvectors of shape y. Shape [B, K, V_y].

    Returns:
        torch.tensor: permutation matrix of shape [B, V_y, V_x].
    """
    Cxy = pointmap2Cxy_vectorized(p2p, evecs_x, evecs_trans_y)
    Pyx = evecs_y @ Cxy @ evecs_trans_x

    return Pyx


def pointmap2Pyx_vectorized(p2p, num_verts_y, num_verts_x):
    """
    Convert point-to-point mapping to the permutation matrix in a vectorized manner.

    Args:
        p2p (torch.Tensor): Point-to-point mapping of shape y -> x. Shape [B, V_y]. _P.S. Invalid points will have value -1._
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
    mask = (p2p != -1) # [B, V_y]
    Pyx = torch.zeros((B, V_y, V_x), dtype=torch.float32, device=p2p.device)
    Pyx[batch_idx[mask], v_y_idx[mask], v_x_idx[mask]] = 1.0

    return Pyx


def corr2pointmap_vectorized(corr_x, corr_y, num_verts_y, spectral_filling: bool = False, evals_x = None, evals_y = None, evecs_x = None, evecs_y = None, evecs_trans_x = None, evecs_trans_y = None, verts_mask_x = None, verts_mask_y = None):
    """
    Convert a pair of ground-truth correspondences to point-to-point map in a vectorized manner.

    **P.S. Every vertex from the template shape will have a corresponding vertex from shape x & y, but not vice versa. Therefore, the established pointmap from y to x is very possible to contain missing points, represented as -1.**

    Args:
        corr_x (torch.Tensor): Correspondences from template to target. Shape [B, V_t] _P.S. V_t is the number of vertices in the template shape._
        corr_y (torch.Tensor): Correspondences from target to template. Shape [B, V_t]
        num_verts_y (torch.Tensor): Number of vertices in the target shape. Shape [B,].
        spectral_filling (bool): Whether to apply spectral filling to the pointmap. If True, the pointmap will be filled with functional maps. If False, the pointmap will be directly constructed from correspondences.
        evals_x (torch.Tensor): _When using spectral filling._ Eigenvalues of shape x. Shape [B, K].
        evals_y (torch.Tensor): _When using spectral filling._ Eigenvalues of shape y. Shape [B, K].
        evecs_x (torch.Tensor): _When using spectral filling._ Eigenvectors of shape x. Shape [B, V_x, K].
        evecs_y (torch.Tensor): _When using spectral filling._ Eigenvectors of shape y. Shape [B, V_y, K].
        evecs_trans_x (torch.Tensor): _When using spectral filling._ Transposed eigenvectors of shape x. Shape [B, K, V_x].
        evecs_trans_y (torch.Tensor): _When using spectral filling._ Transposed eigenvectors of shape y. Shape [B, K, V_y].
        verts_mask_x (torch.Tensor): _When using spectral filling._ Mask for vertices in shape x. Shape [B, V_x] with valid points as 1 and padded points as 0.
        verts_mask_y (torch.Tensor): _When using spectral filling._ Mask for vertices in shape y. Shape [B, V_y] with valid points as 1 and padded points as 0
    Returns:
        p2p (torch.Tensor): Point-to-point map (shape y -> shape x). Shape [B, V_y]  _P.S. Invalid points will have value -1._
    """
    # template -(corr_x)-> shape x <--> shape y <-(corr_y)- target
    # i.e. the i-th row of corr_y is correspongding with the i-th row of corr_x
    B, V_t = corr_x.shape
    batch_idx = torch.arange(B, device=corr_y.device).unsqueeze(1).expand(B, V_t)  # [B, V_t]
    p2p_t = torch.full((B, V_t), -1, dtype=torch.long).to(device=corr_y.device)
    p2p_t[batch_idx, corr_y] = corr_x

    # get p2p in shape [B, V_y]
    V_y = max(num_verts_y)

    if V_t > V_y:
        p2p = p2p_t[:, :V_y]

    else:
        p2p = torch.full((B, V_y), -1, dtype=torch.long).to(device=corr_y.device)
        p2p[:, :V_t] = p2p_t
    
    # spectral filling
    if spectral_filling:
        Cxy_fill = corr2fmap_vectorized(corr_x, corr_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)
        p2p_fill = fmap2pointmap_vectorized(
            Cxy=Cxy_fill,
            evecs_x=evecs_x,
            evecs_y=evecs_y,
            verts_mask_x=verts_mask_x,
            verts_mask_y=verts_mask_y,
        )
        batch_idx, y_no_corr = torch.where(p2p == -1)
        p2p[batch_idx, y_no_corr] = p2p_fill[batch_idx, y_no_corr]

    return p2p


def corr2fmap_vectorized(corr_x, corr_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
    """
    Convert a pair of ground-truth correspondences to point-to-point map which is filled by solving the functional maps problem.

    **P.S. Although the y to x correspondence may be incomplete, they can be converted to corresponding point indicator functions. Then a filled map is solve with functional maps.**

    Args:
        corr_x (torch.Tensor): Correspondences from template to target. Shape [B, V_t] _P.S. V_t is the number of vertices in the template shape._
        corr_y (torch.Tensor): Correspondences from target to template. Shape [B, V_t]
        num_verts_y (int): Number of vertices in the target shape. _P.S. Shared among the whole batch._
        evals_x (torch.Tensor): Eigenvalues of shape x. Shape [B, K]
        evals_y (torch.Tensor): Eigenvalues of shape y. Shape [B, K]
        evecs_trans_x (torch.Tensor): Transposed eigenvectors of shape x. Shape [B, K, V_x]
        evecs_trans_y (torch.Tensor): Transposed eigenvectors of shape y. Shape [B, K, V_y]
    Returns:
        Cxy (torch.Tensor): Functional maps from shape x -> shape y. Shape [B, K, K]
    """
    B, _, V_x = evecs_trans_x.shape
    _, _, V_y = evecs_trans_y.shape
    _, V_t = corr_x.shape

    # point indicator functions
    delta_x = torch.eye(V_x, V_x).repeat(B, 1, 1).to(device=corr_x.device) # [B, V_x, V_x]
    delta_y = torch.eye(V_y, V_y).repeat(B, 1, 1).to(device=corr_y.device) # [B, V_y, V_y]

    # get V_t corresponding point indicator functions as features
    batch_idx = torch.arange(B, device=corr_y.device).unsqueeze(1).expand(B, V_t)  # [B, V_t]
    feat_x = delta_x[batch_idx, corr_x].transpose(1, 2)  # [B, V_t, V_x] -> [B, V_x, V_t]
    feat_y = delta_y[batch_idx, corr_y].transpose(1, 2)  # [B, V_t, V_y] -> [B, V_y, V_t]
    
    # solve the functional map
    fmap_solver = RegularizedFMNet_vectorized(lmbda=0)
    Cxy, _ = fmap_solver(
        feat_x=feat_x,
        feat_y=feat_y,
        evals_x=evals_x,
        evals_y=evals_y,
        evecs_trans_x=evecs_trans_x,
        evecs_trans_y=evecs_trans_y,
    )
    return Cxy