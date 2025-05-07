import os
import torch
from torch import nn

from src.tool import visual
from src.tool.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
class PedarHeatmap(nn.Module):
    def __init__(
            self,
            opt,
            mask_path: str = 'data/processed/left_foot_mask.png',
            sense_range: float = 600,
            export_folder: str = 'sample',
            ):
        super().__init__()
        self.opt = opt
        self.mask_path = mask_path
        self.sense_range = sense_range
        self.export_folder = export_folder

        # check if the export directory exists
        # if not, create it
        export_dir = f'{self.opt.path}/{self.export_folder}'
        
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, dataset_name: str, batch: int):
        # export prediction samples
        for idx, y in enumerate(y_pred.reshape(-1, 198)):
            visual.draw_heatmap(
                arr=y.cpu().detach().numpy() * self.sense_range,
                l_mask_path=self.mask_path,
                is_show=False,
                is_export=True,
                export_path=f'{self.opt.path}/{self.export_folder}/{dataset_name}-{batch}-{idx}-pred.png',
                )
        
        # export groud-truth samples
        for idx, y in enumerate(y_true.reshape(-1, 198)):
            visual.draw_heatmap(
                arr=y.cpu().detach().numpy() * self.sense_range,
                l_mask_path=self.mask_path,
                is_show=False,
                is_export=True,
                export_path=f'{self.opt.path}/{self.export_folder}/{dataset_name}-{batch}-{idx}-true.png',
                )