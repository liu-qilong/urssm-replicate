import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.tool.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class Footprint2Pressure(Dataset):
    def __init__(
            self: str,
            device: str,
            footprint_wrap_folder: str = 'data/processed/footprint-wrap',
            pedar_dynamic_path: str = 'data/processed/pedar_dynamic.pkl',
            l_mask_path: str = 'data/processed/left_foot_mask.png',
            sense_range: float = 600,  # kPa
            stack_range: int = 50,
            img_size: int = 10,
            dtype = torch.float32,
            ):
        self.device = device

        self.footprint_wrap_folder = Path(footprint_wrap_folder)
        self.pedar_dynamic = pd.read_pickle(pedar_dynamic_path)
        self.dtype = dtype
        self.sense_range = sense_range
        self.stack_range = stack_range
        self.img_size = img_size

        self.load_foot_mask(l_mask_path)

        # get index
        self.index = []

        for material, subject in self.pedar_dynamic.index:
            if os.path.isfile(self.footprint_wrap_folder / f'{subject}-L.jpg'):
                self.index.append((material, subject))

    def load_foot_mask(self, l_mask_path: str):
        # load foot masks
        l_img = Image.open(l_mask_path)
        r_img = ImageOps.mirror(l_img)

        self.l_mask = np.array(l_img).astype(np.float64)
        self.r_mask = np.array(r_img).astype(np.float64)

        # detect pixels of area no.1~197 and store the corresponding indexes
        self.l_index = {}
        self.r_index = {}

        for n in range(0, 99):
            self.l_index[n] = np.where(self.l_mask == n + 1)
            self.r_index[n + 99] = np.where(self.r_mask == n + 1)

        # index grids for slicing footprint image as sensor stacks
        range_half = int(self.stack_range / 2)

        self.x_grid = {'L': [], 'R': []}
        self.y_grid = {'L': [], 'R': []}

        for sensor in range(99):
            x_center, y_center = int(self.l_index[sensor][0].mean()), int(self.l_index[sensor][1].mean())
            xs = np.arange(x_center - range_half, x_center + range_half)
            ys = np.arange(y_center - range_half, y_center + range_half)
            xg, yg = np.meshgrid(xs, ys, indexing='ij')
            self.x_grid['L'].append(xg)
            self.y_grid['L'].append(yg)

        for sensor in range(99, 198):
            x_center, y_center = int(self.r_index[sensor][0].mean()), int(self.r_index[sensor][1].mean())
            xs = np.arange(x_center - range_half, x_center + range_half)
            ys = np.arange(y_center - range_half, y_center + range_half)
            xg, yg = np.meshgrid(xs, ys, indexing='ij')
            self.x_grid['R'].append(xg)
            self.y_grid['R'].append(yg)

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx: int) -> tuple:
        # get subject
        material = self.index[idx][0]
        subject = self.index[idx][1]
        
        # get young modulus & pedar arrays
        arr_pedar = self.pedar_dynamic.loc[material, subject].values / self.sense_range
        pedar_t = torch.tensor(arr_pedar, dtype=self.dtype)

        # load and stack left & right footprint images as (2, size, size) tensor
        def get_img_stack(foot: str):
            img = Image.open(self.footprint_wrap_folder / f'{subject}-{foot}.jpg')
            img_arr = np.mean(1 - np.array(img).astype(np.float64) / 255, axis=-1)
            return torch.tensor(img_arr, dtype=self.dtype).unsqueeze(0)
        
        img_l = get_img_stack('L')
        img_r = get_img_stack('R')
        img_stack = torch.concat([img_l, img_r])

        # remember to move data to device!
        return (img_stack.to(self.device), material), pedar_t.to(self.device)


@DATASET_REGISTRY.register()
class Footprint2Pressure_Blend(Footprint2Pressure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # youngs modulus (MPa)
        self.material_youngs = {
            'Poron': 0.33,
            'PElite': 1.11,
            'Lunalight': 5.88,
            'Lunalastic': 0.71,
            'BF': 0.00,
        }

        # get index
        self.index = []

        for subject in self.pedar_dynamic.index.get_level_values(1).drop_duplicates():
            if os.path.isfile(self.footprint_wrap_folder / f'{subject}-L.jpg'):
                self.index.append(subject)
    
    def __getitem__(self, idx: int, blend_weight: np.array = None) -> tuple:
        subject = self.index[idx]

        # blend weights
        if blend_weight is None:
            blend_weight = np.random.rand(5)
            blend_weight = blend_weight / blend_weight.sum()
        
        # weight blends young modulus & pedar arrays
        arr_pedar = self.pedar_dynamic.loc[:, subject, :].values / self.sense_range
        blend_pedar = torch.tensor(
            (arr_pedar * np.expand_dims(blend_weight, axis=-1)).sum(axis=0),
            dtype=self.dtype,
            )
        blend_young = torch.tensor(
            (np.array(list(self.material_youngs.values())) * blend_weight).sum(),
            dtype=self.dtype,
            )

        # load and stack left & right footprint images as (2, size, size) tensor
        def get_img_stack(foot: str):
            img = Image.open(self.footprint_wrap_folder / f'{subject}-{foot}.jpg')
            img_arr = np.mean(1 - np.array(img).astype(np.float64) / 255, axis=-1)
            return torch.tensor(img_arr, dtype=self.dtype).unsqueeze(0)
        
        img_l = get_img_stack('L')
        img_r = get_img_stack('R')
        img_stack = torch.concat([img_l, img_r])

        # remember to move data to device!
        return (img_stack.to(self.device), blend_young.to(self.device)), blend_pedar.to(self.device)
    

@DATASET_REGISTRY.register()
class Footprint2Pressure_SensorStack(Footprint2Pressure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # youngs modulus (MPa)
        self.material_youngs = {
            'Poron': 0.33,
            'PElite': 1.11,
            'Lunalight': 5.88,
            'Lunalastic': 0.71,
            'BF': 0.00,
        }

        self.resize = transforms.Resize((self.img_size, self.img_size))

    def __getitem__(self, idx: int) -> tuple:
        # get subject
        material = self.index[idx][0]
        young = torch.tensor(self.material_youngs[material], dtype=self.dtype)
        subject = self.index[idx][1]
        
        # get young modulus & pedar arrays
        arr_pedar = self.pedar_dynamic.loc[material, subject].values / self.sense_range
        pedar_t = torch.tensor(arr_pedar, dtype=self.dtype)

        # load footprint image and slice as per-sensor stacks
        def get_img_stack(foot: str):
            img = Image.open(self.footprint_wrap_folder / f'{subject}-{foot}.jpg')
            img_arr = np.mean(1 - np.array(img).astype(np.float64) / 255, axis=-1)
            img_stack = img_arr[self.x_grid[foot], self.y_grid[foot]]
            img_stack = torch.tensor(img_stack, dtype=self.dtype)
            img_stack = self.resize(img_stack)
            return img_stack
        
        l_stack = get_img_stack('L')
        r_stack = get_img_stack('R')
        img_stack = torch.concat([l_stack, r_stack])

        # remember to move data to device!
        return (img_stack.to(self.device), young.to(self.device)), pedar_t.to(self.device)


@DATASET_REGISTRY.register()
class Footprint2Pressure_SensorStack_Blend(Footprint2Pressure_Blend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resize = transforms.Resize((self.img_size, self.img_size))

    def __getitem__(self, idx: int, blend_weight: np.array = None) -> tuple:
        if blend_weight is None:
            blend_weight = np.random.rand(5)
            blend_weight = blend_weight / blend_weight.sum()

        # get subject
        subject = self.index[idx]
        
        # weight blends young modulus & pedar arrays
        arr_pedar = self.pedar_dynamic.loc[:, subject, :].values / self.sense_range
        blend_pedar = torch.tensor(
            (arr_pedar * np.expand_dims(blend_weight, axis=-1)).sum(axis=0),
            dtype=self.dtype,
            )
        blend_young = torch.tensor(
            (np.array(list(self.material_youngs.values())) * blend_weight).sum(),
            dtype=self.dtype,
            )

        # load footprint image and slice as per-sensor stacks
        def get_img_stack(foot: str):
            img = Image.open(self.footprint_wrap_folder / f'{subject}-{foot}.jpg')
            img_arr = np.mean(1 - np.array(img).astype(np.float64) / 255, axis=-1)
            img_stack = img_arr[self.x_grid[foot], self.y_grid[foot]]
            img_stack = torch.tensor(img_stack, dtype=self.dtype)
            img_stack = self.resize(img_stack)
            return img_stack
        
        l_stack = get_img_stack('L')
        r_stack = get_img_stack('R')
        img_stack = torch.concat([l_stack, r_stack])

        # remember to move data to device!
        return (img_stack.to(self.device), blend_young.to(self.device)), blend_pedar.to(self.device)
    

@DATASET_REGISTRY.register()
class Footprint2Pressure_SensorPatch_Blend(Footprint2Pressure_Blend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # transformer for resizing images
        self.resize = transforms.Resize((self.img_size, self.img_size))

        # data indexing and footprint image preloading
        self.index = []
        self.subject2img_stack = {}

        for subject in self.pedar_dynamic.index.get_level_values(1).drop_duplicates():
            if os.path.isfile(self.footprint_wrap_folder / f'{subject}-L.jpg') and os.path.isfile(self.footprint_wrap_folder / f'{subject}-R.jpg'):
                for patch_id in range(1, 199):
                    self.index.append((subject, patch_id))

                # pre-load and slice the footprint images
                def get_img_stack(foot: str):
                    img = Image.open(self.footprint_wrap_folder / f'{subject}-{foot}.jpg')
                    img_arr = np.mean(1 - np.array(img).astype(np.float64) / 255, axis=-1)
                    img_stack = img_arr[self.x_grid[foot], self.y_grid[foot]]
                    img_stack = torch.tensor(img_stack, dtype=self.dtype)
                    img_stack = self.resize(img_stack)
                    return img_stack
                
                l_stack = get_img_stack('L')  # e.g. (99, 10, 10)
                r_stack = get_img_stack('R')  # e.g. (99, 10, 10)
                img_stack = torch.concat([l_stack, r_stack])  # e.g. (198, 10, 10)
                self.subject2img_stack[subject] = img_stack
    
    def __getitem__(self, idx: int, blend_weight: np.array = None) -> tuple:
        subject, sensor_id = self.index[idx]
        
        # blend weights
        if blend_weight is None:
            blend_weight = np.random.rand(5)
            blend_weight = blend_weight / blend_weight.sum()

        # weight blends young modulus & pedar arrays
        arr_pedar = self.pedar_dynamic.loc[(slice(None), subject), sensor_id].values / self.sense_range
        blend_pedar = torch.tensor(
            (arr_pedar * blend_weight).sum(axis=0),
            dtype=self.dtype,
            )
        blend_young = torch.tensor(
            (np.array(list(self.material_youngs.values())) * blend_weight).sum(),
            dtype=self.dtype,
            )
        
        # get sensor-specific patch
        img_patch = self.subject2img_stack[subject][sensor_id - 1]  # sensor_id starts from 1
        
        return (img_patch.to(self.device), torch.tensor(sensor_id, dtype=self.dtype).to(self.device), blend_young.to(self.device)), blend_pedar.to(self.device)