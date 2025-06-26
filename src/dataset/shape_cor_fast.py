import os
from glob import glob
from pathlib import Path
from itertools import product

import torch
import numpy as np
from torch.utils.data import Dataset

from src.infra.registry import DATASET_REGISTRY
from src.utils.tensor import sparse_np_to_torch, read_sp_mat

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

        item['name'] = off_fname

        spectral_npz = np.load(
            self.data_root / 'spectral' / f'{off_fname}.npz'
        )

        assert spectral_npz['k_eig'] >= self.num_evecs, 'not enough eigenvectors in spectral data'

        # get vertices and faces
        item['verts'] = torch.from_numpy(spectral_npz['verts'])

        if self.return_faces:
            item['faces'] = torch.from_numpy(spectral_npz['faces'])

        if self.return_L:
            item['L'] = sparse_np_to_torch(read_sp_mat(spectral_npz, 'L'))
            
        if self.return_mass:
            item['mass'] = torch.from_numpy(spectral_npz['mass'])

        if self.return_evecs:
            item['evecs'] = torch.from_numpy(spectral_npz['evecs'][:, :self.num_evecs])
            item['evecs_trans'] = torch.from_numpy(
                spectral_npz['evecs'][:, :self.num_evecs].T * spectral_npz['mass'][None]
            )  # p.s. this step could take around 1/3 of the __getittem__ runtime. could be optimized by precomputation
            item['evals'] = torch.from_numpy(spectral_npz['evals'][:self.num_evecs])

        if self.return_grad:
            item['gradX'] = sparse_np_to_torch(read_sp_mat(spectral_npz, 'gradX'))
            item['gradY'] = sparse_np_to_torch(read_sp_mat(spectral_npz, 'gradY'))
        
        if self.return_corr:
            item['corr'] = torch.from_numpy(np.loadtxt(
                self.data_root / 'corres' / f'{off_fname}.vts',
                dtype=np.int32,
            ) - 1)  # minus 1 to start from 0

        if self.return_dist:
            item['dist'] = torch.from_numpy(np.load(
                self.data_root / 'dist' / f'{off_fname}.npz'
            )['dist_mat'])

        return item

    def __len__(self):
        return len(self.off_files)
    

@DATASET_REGISTRY.register()
class FaustDatasetFast(ShapeDatasetFast):
    def __init__(
            self,
            data_root,
            phase,
            return_faces=True,
            return_L=False,
            return_mass=True,
            num_evecs=200,
            return_evecs=True,
            return_grad=True,
            return_corr=False,
            return_dist=False,
        ):
        super(FaustDatasetFast, self).__init__(
            data_root,
            return_faces,
            return_L,
            return_mass,
            num_evecs,
            return_evecs,
            return_grad,
            return_corr,
            return_dist,
        )

        assert phase in ['train', 'test', 'full'], f'Invalid phase {phase}, only "train" or "test" or "full"'
        assert len(self) == 100, f'FAUST dataset should contain 100 human body shapes, but get {len(self)}.'

        if phase == 'train':
            self.off_files = self.off_files[:80]

        elif phase == 'test':
            self.off_files = self.off_files[80:]


class PairShapeDataset(Dataset):
    def __init__(self, dataset):
        """
        Pair Shape Dataset

        Args:
            dataset (SingleShapeDataset): single shape dataset
        """
        assert isinstance(dataset, ShapeDatasetFast), f'invalid input data type of dataset: {type(dataset)}'
        self.dataset = dataset
        self.combinations = list(product(range(len(dataset)), repeat=2))

    def __getitem__(self, index):
        # get index
        first_index, second_index = self.combinations[index]

        item = dict()
        item['first'] = self.dataset[first_index]
        item['second'] = self.dataset[second_index]

        return item

    def __len__(self):
        return len(self.combinations)


@DATASET_REGISTRY.register()
class PairFaustDatasetFast(PairShapeDataset):
    def __init__(
            self,
            data_root,
            phase,
            return_faces=True,
            return_L=False,
            return_mass=True,
            num_evecs=200,
            return_evecs=True,
            return_grad=True,
            return_corr=False,
            return_dist=False,
        ):
        dataset = FaustDatasetFast(
            data_root,
            phase,
            return_faces,
            return_L,
            return_mass,
            num_evecs,
            return_evecs,
            return_grad,
            return_corr,
            return_dist,
        )
        super(PairFaustDatasetFast, self).__init__(dataset)