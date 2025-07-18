import time
import shutil
from pathlib import Path
import pyvista as pv

import torch
import torch.distributed as dist

from src.metric import BaseMetric
from src.infra.registry import METRIC_REGISTRY
from src.utils.fmap import fmap2pointmap_vectorized
from src.utils.fmap import pointmap2Pyx_vectorized
from src.utils.tensor import to_numpy
from src.utils.texture import write_obj_pair

@METRIC_REGISTRY.register()
class TextureTransferSample(BaseMetric):
    """Generate texture transfer samples.
    """
    def __init__(self, batch_interval: int = 10, texture_file: str = 'gallery/texture-symbol-grid.png', shape_disp: list = [1, 0, 0], cammera_position: str = 'xy', window_size: list = [1024, 1024], output_folder: str = 'bench/texture-transfer-samples', keep_mesh: bool = True):
        """
        Args:
            texture_file (str): Path to the texture file.
            shape_disp (list): Displacement of the second shape.
            cammera_position (str): Position of the camera.
            batch_interval (int): Generate texture transfer samples per `batch_interval` batches.
            output_folder (str): Folder to save the generated samples (relative to the experiment folder)
        """
        super().__init__()
        self.batch_interval = batch_interval
        self.texture_file = texture_file
        self.shape_disp = shape_disp
        self.cammera_position = cammera_position
        self.window_size = window_size
        self.output_folder = output_folder
        self.keep_mesh = keep_mesh

    def start_feed(self, script, name, rank=None):
        """Log start time
        
        Args:
            script: The traning/benchmark script object
            name: The name of the metric method
            rank: The rank of the process (if using distributed training)
        """
        super().start_feed(script, name, rank)
        self.batch_total = 0

        # create output folder
        self.output_path = Path(script.opt.path) / self.output_folder
        self.output_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.texture_file, self.output_path / 'texture.png')

    def forward(self, infer, data):
        """Generate texture transfer samples per `batch_interval` batches.
        """
        if self.batch_total % self.batch_interval == 0:
            p2p = fmap2pointmap_vectorized(
                infer['Cxy'],
                data['first']['evecs'], data['second']['evecs'],
                data['first']['verts_mask'], data['second']['verts_mask']
            )

            Pyx = pointmap2Pyx_vectorized(
                p2p=p2p,
                evecs_x = data['first']['evecs'],
                evecs_y = data['second']['evecs'],
                evecs_trans_x = data['first']['evecs_trans'],
                evecs_trans_y = data['second']['evecs_trans'],
            )

            name_x, name_y = data['first']['name'], data['second']['name']
            verts_num_x = data['first']['num_verts']
            verts_num_y = data['second']['num_verts']
            faces_num_x = data['first']['num_faces']
            faces_num_y = data['second']['num_faces']

            for idx in range(len(infer['Cxy'])):
                # export mesh w/ texture
                write_obj_pair(
                    file_name1=str(self.output_path / f'{name_x[idx]}.obj'),
                    file_name2=str(self.output_path / f'{name_x[idx]}--{name_y[idx]}.obj'),
                    faces1=to_numpy(data['first']['faces'][idx, :faces_num_x[idx]]),
                    verts1=to_numpy(data['first']['verts'][idx, :verts_num_x[idx]]),
                    verts2=to_numpy(data['second']['verts'][idx, :verts_num_y[idx]]),
                    faces2=to_numpy(data['second']['faces'][idx, :faces_num_y[idx]]),
                    Pyx=to_numpy(Pyx[idx, :verts_num_y[idx], :verts_num_x[idx]]),
                    texture_file=str(self.output_path / 'texture.png'),
                )

                # render texture transfer
                pl = pv.Plotter(off_screen=True)
                pl.add_mesh(
                    mesh=pv.read(self.output_path / f'{name_x[idx]}.obj'),
                    texture=pv.read_texture(self.output_path / 'texture.png'),
                )
                pl.add_mesh(
                    mesh=pv.read(self.output_path / f'{name_x[idx]}--{name_y[idx]}.obj').translate(self.shape_disp),
                    texture=pv.read_texture(self.output_path / 'texture.png'),
                )
                pl.camera_position = 'xy'
                pl.screenshot(self.output_path / f'{name_x[idx]}--{name_y[idx]}.png', window_size=self.window_size, return_img=False)
                pl.close()

                # remove the mesh files if not needed
                if not self.keep_mesh:
                    (self.output_path / f'{name_x[idx]}.obj').unlink(missing_ok=True)
                    (self.output_path / f'{name_x[idx]}.mtl').unlink(missing_ok=True)
                    (self.output_path / f'{name_x[idx]}--{name_y[idx]}.obj').unlink(missing_ok=True)
                    (self.output_path / f'{name_x[idx]}--{name_y[idx]}.mtl').unlink(missing_ok=True)

        # increment batch total
        self.batch_total += 1
        
        # return zero
        return torch.tensor(0.0).to(device=self.script.device)