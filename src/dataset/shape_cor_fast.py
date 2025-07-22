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
            mesh_type: str = 'off',
            return_faces: bool =True,
            return_L: bool = False,
            return_mass: bool = True,
            num_evecs: int = 200,
            return_evecs: bool = True,
            return_grad: bool = True,
            return_corr: bool = False,
            return_dist: bool = False,
        ):

        # options
        self.data_root = Path(data_root)
        assert os.path.isdir(data_root), f'invalid data root: {data_root}.'

        self.mesh_type = mesh_type
        self.return_faces = return_faces
        self.return_evecs = return_evecs
        self.return_grad = return_grad
        self.return_L = return_L
        self.return_mass = return_mass
        self.return_corr = return_corr
        self.return_dist = return_dist
        self.num_evecs = num_evecs

        self.mesh_files = sorted(glob(str(self.data_root / self.mesh_type / f'*.{self.mesh_type}')))

        # sanity checks
        assert len(self.mesh_files)

        if self.return_dist:
            assert os.path.isdir(self.data_root / 'dist'), f'dist folder not found: {self.data_root / "dist"}'

        if self.return_corr:
            assert os.path.isdir(self.data_root / 'corres'), f'corres folder not found: {self.data_root / "corres"}'


    def __getitem__(self, index):
        item = {}

        # get shape name and load spectral npz
        mesh_fname = Path(self.mesh_files[index]).stem

        item['name'] = mesh_fname

        spectral_npz = np.load(
            self.data_root / 'spectral' / f'{mesh_fname}.npz'
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
                self.data_root / 'corres' / f'{mesh_fname}.vts',
                dtype=np.int32,
            ) - 1).long()  # minus 1 to start from 0

        if self.return_dist:
            item['dist'] = torch.from_numpy(np.load(
                self.data_root / 'dist' / f'{mesh_fname}.npz'
            )['dist_mat'])

        return item

    def __len__(self):
        return len(self.mesh_files)
    

@DATASET_REGISTRY.register()
class FaustDatasetFast(ShapeDatasetFast):
    def __init__(
            self,
            data_root,
            phase,
            mesh_type='off',
            return_faces=True,
            return_L=False,
            return_mass=True,
            num_evecs=200,
            return_evecs=True,
            return_grad=True,
            return_corr=False,
            return_dist=False,
        ):
        super().__init__(
            data_root,
            mesh_type,
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
            self.mesh_files = self.mesh_files[:80]

        elif phase == 'test':
            self.mesh_files = self.mesh_files[80:]


@DATASET_REGISTRY.register()
class ScapeDatasetFast(ShapeDatasetFast):
    def __init__(
            self,
            data_root,
            phase,
            mesh_type='off',
            return_faces=True,
            return_L=False,
            return_mass=True,
            num_evecs=200,
            return_evecs=True,
            return_grad=True,
            return_corr=False,
            return_dist=False,
        ):
        super().__init__(
            data_root,
            mesh_type,
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
            self.mesh_files = self.mesh_files[:51]

        elif phase == 'test':
            self.mesh_files = self.mesh_files[51:]


@DATASET_REGISTRY.register()
class Shrec16DatasetFast(ShapeDatasetFast):
    def __init__(
            self,
            data_root,
            phase,
            mesh_type='off',
            return_faces=True,
            return_L=False,
            return_mass=True,
            num_evecs=200,
            return_evecs=True,
            return_grad=True,
            return_corr=False,
            return_dist=False,
        ):
        super().__init__(
            data_root,
            mesh_type,
            return_faces,
            return_L,
            return_mass,
            num_evecs,
            return_evecs,
            return_grad,
            return_corr,
            return_dist,
        )
        if phase == 'train':
            assert '_test' not in str(self.data_root), f'Invalid data root {self.data_root}, should be SHREC16/<cls>'

        elif phase == '_test':
            assert 'test' in str(self.data_root), f'Invalid data root {self.data_root}, should be SHREC16_test/<cls>'

        else:
            raise ValueError(f'Invalid phase {phase}, only "train" or "test"')


@DATASET_REGISTRY.register()
class Shrec19DatasetFast(ShapeDatasetFast):
    def __init__(
            self,
            data_root,
            phase,
            mesh_type='off',
            return_faces=True,
            return_L=False,
            return_mass=True,
            num_evecs=200,
            return_evecs=True,
            return_grad=True,
            return_corr=False,
            return_dist=False,
        ):
        super().__init__(
            data_root,
            mesh_type,
            return_faces,
            return_L,
            return_mass,
            num_evecs,
            return_evecs,
            return_grad,
            return_corr,
            return_dist,
        )
        assert phase == 'test', f'Shrec19 dataset is only used as test set'
       

@DATASET_REGISTRY.register()
class Shrec20DatasetFast(ShapeDatasetFast):
    def __init__(
            self,
            data_root,
            mesh_type='off',
            return_faces=True,
            return_L=False,
            return_mass=True,
            num_evecs=200,
            return_evecs=True,
            return_grad=True,
            return_corr=False,
            return_dist=False,
        ):
        assert not return_corr, 'Shrec20 dataset does not contain ground-truth correspondence.'
        assert not return_dist, 'Shrec20 dataset does not contain ground-truth correspondence. Do not set return_dist to True.'
        super().__init__(
            data_root,
            mesh_type,
            return_faces,
            return_L,
            return_mass,
            num_evecs,
            return_evecs,
            return_grad,
            return_corr,
            return_dist,
        )


@DATASET_REGISTRY.register()
class TopKidsDatasetFast(ShapeDatasetFast):
    def __init__(
            self,
            data_root,
            mesh_type='off',
            return_faces=True,
            return_L=False,
            return_mass=True,
            num_evecs=200,
            return_evecs=True,
            return_grad=True,
            return_corr=False,
            return_dist=False,
        ):
        assert not return_corr, 'TopKids dataset does not contain ground-truth correspondence.'
        assert not return_dist, 'TopKids dataset does not contain ground-truth correspondence. Do not set return_dist to True.'
        super().__init__(
            data_root,
            mesh_type,
            return_faces,
            return_L,
            return_mass,
            num_evecs,
            return_evecs,
            return_grad,
            return_corr,
            return_dist,
        )


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
            mesh_type='off',
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
            mesh_type,
            return_faces,
            return_L,
            return_mass,
            num_evecs,
            return_evecs,
            return_grad,
            return_corr,
            return_dist,
        )
        super().__init__(dataset)