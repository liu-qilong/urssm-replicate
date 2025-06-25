import os
from glob import glob
from pathlib import Path

import scipy
import numpy as np
from torch.utils.data import Dataset

from src.infra.registry import DATASET_REGISTRY

def read_sp_mat(npz, prefix):
    data = npz[prefix + '_data']
    indices = npz[prefix + '_indices']
    indptr = npz[prefix + '_indptr']
    shape = npz[prefix + '_shape']
    mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
    return mat

@DATASET_REGISTRY.register()
class ShapeDatasetFast(Dataset):
    def __init__(
            self,
            data_root,
            return_faces=True,
            return_L=False,
            return_mass=True,
            num_evecs=200,
            return_evecs=True,
            return_grad=True,
            return_corr=False,
            return_dist=False,
        ):

        # options
        self.data_root = Path(data_root)
        assert os.path.isdir(data_root), f'invalid data root: {data_root}.'

        self.return_faces = return_faces
        self.return_evecs = return_evecs
        self.return_grad = return_grad
        self.return_L = return_L
        self.return_mass = return_mass
        self.return_corr = return_corr
        self.return_dist = return_dist
        self.num_evecs = num_evecs

        self.off_files = sorted(glob(str(self.data_root / 'off' / '*.off')))

        # sanity checks
        assert len(self.off_files)

        if self.return_dist:
            assert os.path.isdir(self.data_root / 'dist'), f'dist folder not found: {self.data_root / "dist"}'

        if self.return_corr:
            assert os.path.isdir(self.data_root / 'corres'), f'corres folder not found: {self.data_root / "corres"}'


    def __getitem__(self, index):
        item = {}

        # get shape name and load spectral npz
        off_fname = Path(self.off_files[index]).stem

        spectral_npz = np.load(
            self.data_root / 'spectral' / f'{off_fname}.npz'
        )

        assert spectral_npz['k_eig'] >= self.num_evecs, 'not enough eigenvectors in spectral data'

        # get vertices and faces
        item['verts'] = spectral_npz['verts']
        
        if self.return_faces:
            item['faces'] = spectral_npz['faces']

        if self.return_L:
            item['L'] = read_sp_mat(spectral_npz, 'L')
            
        if self.return_mass:
            item['mass'] = spectral_npz['mass']

        if self.return_evecs:
            item['evecs'] = spectral_npz['evecs'][:, :self.num_evecs]
            item['evals'] = spectral_npz['evals'][:self.num_evecs]

        if self.return_grad:
            item['gradX'] = read_sp_mat(spectral_npz, 'gradX')
            item['gradY'] = read_sp_mat(spectral_npz, 'gradY')
        
        if self.return_corr:
            item['corr'] = np.loadtxt(
                self.data_root / 'corres' / f'{off_fname}.vts',
                dtype=np.int32,
            ) - 1  # minus 1 to start from 0

        if self.return_dist:
            item['dist'] = np.load(
                self.data_root / 'dist' / f'{off_fname}.npz'
            )

        return item

    def __len__(self):
        return len(self.off_files)