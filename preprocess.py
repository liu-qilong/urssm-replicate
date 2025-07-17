# This implementation is adapted from Dongliang Cao, et al. (2024): https://github.com/dongliangcao/unsupervised-learning-of-robust-spectral-shape-matching. New modules & refinements are added

import os
from pathlib import Path
from argparse import ArgumentParser
from glob import glob

import torch
import open3d as o3d
import numpy as np
from tqdm.auto import tqdm

from src.utils.geometry import compute_operators, torch2np, sparse_torch_to_np
from src.utils.shape import compute_geodesic_distmat


if __name__ == '__main__':
    # parse arguments
    parser = ArgumentParser('preprocess mesh dataset')
    parser.add_argument('--data_root', required=True, help='data root contains a sub-folder named as <mesh_type>/ containing the mesh files.')
    parser.add_argument('--mesh_type', required=False, default='off', help='mesh file type (e.g. off, ply, obj).')
    parser.add_argument('--k_eig', type=int, default=128, help='number of eigenvectors/values to compute.')
    parser.add_argument('--no_dist', action='store_true', help='no geodesic matrix.')
    args = parser.parse_args()

    # params
    data_root = Path(args.data_root)
    mesh_type = args.mesh_type
    no_dist = args.no_dist
    k_eig = args.k_eig
    assert k_eig > 0, f'invalid k_eig: {k_eig}'
    assert os.path.isdir(data_root), f'invalid data root: {data_root}'

    spectral_dir = data_root / 'spectral'
    os.makedirs(spectral_dir, exist_ok=True)

    # geodist
    if not no_dist:
        dist_dir = data_root / 'dist'
        os.makedirs(dist_dir, exist_ok=True)

    # preprocessing loop
    mesh_files = sorted(glob(str(data_root / mesh_type / f'*.{mesh_type}')))
    assert len(mesh_files) != 0
    
    for mesh_file in tqdm(mesh_files):
        # load mesh
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)

        # lbo
        frames, mass_vec, L, evals, evecs, gradX, gradY = compute_operators(
            torch.from_numpy(verts).float(),
            torch.from_numpy(faces).long(),
            k=k_eig,
        )

        # save to npz (w/ verts & faces)
        frames_np = torch2np(frames).astype(np.float32)
        mass_np = torch2np(mass_vec).astype(np.float32)
        evals_np = torch2np(evals).astype(np.float32)
        evecs_np = torch2np(evecs).astype(np.float32)
        L_np = sparse_torch_to_np(L).astype(np.float32)
        gradX_np = sparse_torch_to_np(gradX).astype(np.float32)
        gradY_np = sparse_torch_to_np(gradY).astype(np.float32)

        np.savez(
            spectral_dir / f'{Path(mesh_file).stem}.npz',
            verts=verts,
            faces=faces,
            k_eig=k_eig,
            frames=frames_np,
            mass=mass_np,
            evals=evals_np,
            evecs=evecs_np,
            L_data=L_np.data,
            L_indices=L_np.indices,
            L_indptr=L_np.indptr,
            L_shape=L_np.shape,
            gradX_data=gradX_np.data,
            gradX_indices=gradX_np.indices,
            gradX_indptr=gradX_np.indptr,
            gradX_shape=gradX_np.shape,
            gradY_data=gradY_np.data,
            gradY_indices=gradY_np.indices,
            gradY_indptr=gradY_np.indptr,
            gradY_shape=gradY_np.shape,
        )

        # geodist
        if not no_dist:
            dist_mat = compute_geodesic_distmat(verts, faces)
            np.savez(
                dist_dir / f'{Path(mesh_file).stem}.npz',
                dist_mat=dist_mat,
            )
