# This implementation is adapted from Dongliang Cao, et al. (2024): https://github.com/dongliangcao/unsupervised-learning-of-robust-spectral-shape-matching. New modules & refinements are added

import scipy
import torch
import numbers
import numpy as np


def _to_device(x, device):
    if torch.is_tensor(x):
        x = x.to(device=device)
    return x


def to_device(x, device):
    if isinstance(x, list):
        x = [to_device(x_i, device) for x_i in x]
        return x
    elif isinstance(x, dict):
        x = {k: to_device(v, device) for (k, v) in x.items()}
        return x
    else:
        return _to_device(x, device)


def to_numpy(tensor, squeeze=True):
    """Wrapper around .detach().cpu().numpy() """
    if isinstance(tensor, torch.Tensor):
        if squeeze:
            tensor = tensor.squeeze()
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, numbers.Number):
        return np.array([tensor])
    else:
        raise NotImplementedError()


def to_tensor(ndarray):
    if isinstance(ndarray, torch.Tensor):
        return ndarray
    elif isinstance(ndarray, np.ndarray):
        return torch.from_numpy(ndarray)
    elif isinstance(ndarray, numbers.Number):
        return torch.tensor(ndarray)
    else:
        raise NotImplementedError()


def to_number(ndarray):
    if isinstance(ndarray, torch.Tensor) or isinstance(ndarray, np.ndarray):
        return ndarray.item()
    elif isinstance(ndarray, numbers.Number):
        return ndarray
    else:
        raise NotImplementedError()


def select_points(pts, idx):
    """
    select points based on given indices
    Args:
        pts (tensor): points [B, N, C]
        idx (tensor): indices [B, M]

    Returns:
        (tensor): selected points [B, M, C]
    """
    selected_pts = torch.stack([pts[i, idx[i]] for i in range(pts.shape[0])], dim=0)
    return selected_pts

def tensor_memory_footprint(x):
    if x.is_sparse:
        values = x._values()
        indices = x._indices()
        num_bytes = values.numel() * values.element_size() + indices.numel() * indices.element_size()
        print(f"Sparse tensor size: {num_bytes / (1024 ** 2):.2f} MB")
    else:
        num_bytes = x.numel() * x.element_size()
        print(f"Dense tensor size: {num_bytes / (1024 ** 2):.2f} MB")

def torch2np(tensor):
    assert isinstance(tensor, torch.Tensor)
    return tensor.detach().cpu().numpy()

def read_sp_mat(npz, prefix):
    data = npz[prefix + '_data']
    indices = npz[prefix + '_indices']
    indptr = npz[prefix + '_indptr']
    shape = npz[prefix + '_shape']
    mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
    return mat

def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)).coalesce()


def sparse_torch_to_np(A):
    assert len(A.shape) == 2

    indices = torch2np(A.indices())
    values = torch2np(A.values())

    mat = scipy.sparse.coo_matrix((values, indices), shape=A.shape).tocsc()
    return mat