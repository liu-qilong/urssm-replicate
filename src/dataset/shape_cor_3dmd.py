from src.dataset.shape_cor_fast import ShapeDatasetFast, PairShapeDataset
from src.infra.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class Single3dMDDatasetFast(ShapeDatasetFast):
    def __init__(
            self,
            data_root,
            mesh_type='ply',
            return_faces=True,
            return_L=False,
            return_mass=True,
            num_evecs=200,
            return_evecs=True,
            return_grad=True,
            return_corr=False,
            return_dist=False,
        ):
        # groudntruth shapr cor unavailable
        assert not return_corr

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
class Pair3dMDDataset(PairShapeDataset):
    def __init__(
            self,
            data_root,
            mesh_type='ply',
            return_faces=True,
            return_L=False,
            return_mass=True,
            num_evecs=200,
            return_evecs=True,
            return_grad=True,
            return_corr=False,
            return_dist=False,
        ):
        dataset = Single3dMDDatasetFast(
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
        super().__init__(dataset)