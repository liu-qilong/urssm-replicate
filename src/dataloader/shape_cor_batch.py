import torch
from torch.utils.data import DataLoader

from src.infra.registry import DATALOADER_REGISTRY

@DATALOADER_REGISTRY.register()
class BatchShapePairDataLoader(DataLoader):
    def __init__(self, dataset, *args, **kwargs):
        # customized collate_fn
        def collate_shape_batch(batch):
            # find maximum number of vertices in the batch
            if 'verts' in batch[0]['first']:
                max_verts_dict = {}
                max_verts_dict['first'] = max([sample['first']['verts'].shape[0] for sample in batch])
                max_verts_dict['second'] = max([sample['second']['verts'].shape[0] for sample in batch])
            
            # find maximum number of faces (if you want to batch faces too)
            if 'faces' in batch[0]['first']:
                max_faces_dict = {}
                max_faces_dict['first'] = max([sample['first']['faces'].shape[0] for sample in batch])
                max_faces_dict['second'] = max([sample['second']['faces'].shape[0] for sample in batch])

            if 'corr' in batch[0]['first']:
                # find maximum number of correspondences
                # p.s. the i-th line with value v means the i-th vectex from the template shape is corresponding to v-th vertex in the current shape
                max_corr_num = max([
                    len(sample[key]['corr'])
                    for sample in batch
                    for key in ['first', 'second']
                ])

            # pad each data sample to the unified shape
            collated_batch = []
            
            for sample in batch:
                collated_sample = {'first': {}, 'second': {}}
                
                for shape_key in ['first', 'second']:
                    shape_data = sample[shape_key]
                    
                    # name
                    collated_sample[shape_key]['name'] = shape_data['name']
                    
                    if 'verts' in shape_data:
                        max_verts = max_verts_dict[shape_key]
                        current_verts = shape_data['verts'].shape[0]
                        collated_sample[shape_key]['num_verts'] = current_verts
                        collated_sample[shape_key]['verts_mask'] = torch.zeros(max_verts)
                        collated_sample[shape_key]['verts_mask'][:current_verts] = 1

                        # pad vertex coordinates (V, 3) -> (max_V, 3)
                        verts_padded = torch.zeros(max_verts, 3)
                        verts_padded[:current_verts] = shape_data['verts']
                        collated_sample[shape_key]['verts'] = verts_padded

                    if 'faces' in shape_data:
                        max_faces = max_faces_dict[shape_key]
                        current_faces = shape_data['faces'].shape[0]
                        collated_sample[shape_key]['num_faces'] = current_faces

                        # pad faces (F, 3) -> (max_F, 3)
                        # note: padded faces will have invalid indices, you might want to set them to -1
                        faces_padded = torch.full((max_faces, 3), -1, dtype=shape_data['faces'].dtype)
                        faces_padded[:current_faces] = shape_data['faces']
                        collated_sample[shape_key]['faces'] = faces_padded

                    if 'L' in shape_data:
                        # pad sparse Laplacian matrix (V, V) -> (max_V, max_V)
                        collated_sample[shape_key]['L'] = torch.sparse_coo_tensor(
                            shape_data['L'].indices(),
                            shape_data['L'].values(),
                            (max_verts, max_verts),
                        )

                    if 'mass' in shape_data:
                        # pad mass vector (V, ) -> (max_V, )
                        mass_padded = torch.zeros(max_verts)
                        mass_padded[:current_verts] = shape_data['mass']
                        collated_sample[shape_key]['mass'] = mass_padded
                    
                    if 'evecs' in shape_data:
                        # pad eigenvectors (V, K) -> (max_V, K)
                        K = shape_data['evecs'].shape[1]
                        evecs_padded = torch.zeros(max_verts, K)
                        evecs_padded[:current_verts] = shape_data['evecs']
                        collated_sample[shape_key]['evecs'] = evecs_padded
                    
                    if 'evecs_trans' in shape_data:
                        # pad transposed eigenvectors (K, V) -> (K, max_V)
                        evecs_trans_padded = torch.zeros(K, max_verts)
                        evecs_trans_padded[:, :current_verts] = shape_data['evecs_trans']
                        collated_sample[shape_key]['evecs_trans'] = evecs_trans_padded
                    
                    if 'evals' in shape_data:
                        # eigenvalues don't need padding (K,)
                        collated_sample[shape_key]['evals'] = shape_data['evals']

                    if 'gradX' in shape_data:
                        # pad sparse gradient X (V, V) -> (max_V, max_V)
                        collated_sample[shape_key]['gradX'] = torch.sparse_coo_tensor(
                            shape_data['gradX'].indices(),
                            shape_data['gradX'].values(),
                            (max_verts, max_verts),
                        )

                    if 'gradY' in shape_data:
                        # pad sparse gradient Y (V, V) -> (max_V, max_V)
                        collated_sample[shape_key]['gradY'] = torch.sparse_coo_tensor(
                            shape_data['gradY'].indices(),
                            shape_data['gradY'].values(),
                            (max_verts, max_verts),
                        )
                    
                    if 'corr' in shape_data:
                        # pad correspondence (V,) -> (max_corr_num,)
                        corr_padded = torch.full((max_corr_num,), -1, dtype=shape_data['corr'].dtype)
                        corr_padded[:len(shape_data['corr'])] = shape_data['corr']
                        collated_sample[shape_key]['corr'] = corr_padded

                    if 'dist' in shape_data:
                        # pad distance matrix (V, V) -> (max_V, max_V)
                        dist_padded = torch.zeros(max_verts, max_verts)
                        dist_padded[:current_verts, :current_verts] = shape_data['dist']
                        collated_sample[shape_key]['dist'] = dist_padded
                
                collated_batch.append(collated_sample)
            
            # stack all samples in the batch
            batched_data = {'first': {}, 'second': {}}
            
            for shape_key in ['first', 'second']:
                for field in collated_batch[0][shape_key]:
                    first_sample = collated_batch[0][shape_key][field]
                    
                    if isinstance(first_sample, torch.Tensor):
                        if first_sample.is_sparse:
                            # stack sparse tensors
                            # p.s. since sparse tensor can't work with dataloader when num_workers > 0,
                            # indices, values, sizes are extracted and stacked separately as dense tensors
                            stack_sparse_tensor = torch.stack([
                                sample[shape_key][field] for sample in collated_batch
                            ]).coalesce()
                            batched_data[shape_key][field + '_indices'] = stack_sparse_tensor.indices()
                            batched_data[shape_key][field + '_values'] = stack_sparse_tensor.values()
                            batched_data[shape_key][field + '_size'] = stack_sparse_tensor.size()

                        else:
                            # stack dense tensors
                            batched_data[shape_key][field] = torch.stack([
                                sample[shape_key][field] for sample in collated_batch
                            ])

                    else:
                        # keep non-tensor fields as lists
                        batched_data[shape_key][field] = [
                            sample[shape_key][field] for sample in collated_batch
                        ]
            
            return batched_data
        
        super().__init__(
            dataset,
            *args,
            collate_fn=collate_shape_batch,
            **kwargs,
        )