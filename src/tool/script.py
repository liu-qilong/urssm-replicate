import torch
from torch.utils.data import random_split
import pandas as pd

import time
from tqdm.auto import tqdm
from pathlib import Path

from src.tool.registry import DATASET_REGISTRY, DATALOADER_REGISTRY, MODEL_REGISTRY, LOSS_REGISTRY, METRIC_REGISTRY, OPTIMIZER_REGISTRY, SCRIPT_REGISTRY

# --- training scripts ---
@SCRIPT_REGISTRY.register()
class TrainScript():
    def __init__(self, opt):
        self.opt = opt
        
        # device select
        if self.opt.device_select == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = self.opt.device_select

        print(f'running on {self.device}...')

        # init logs dict
        self.logs = {
            'epoch': [],
            'time': [],
            'train_loss': [],
            'test_loss': [],
        }

        # init metric dict
        self.metric_dict = {}

        for key, value in self.opt.train.metric.items():
            self.metric_dict[key] = METRIC_REGISTRY[value.name](**value.args)
            self.logs[key] = []

    def load_data(self):
        # to enforce consistency of train/test dataset splitting, the train/test set shall be explicitly provided as different dataset, rather than created with random split.
        self.train_dataset = DATASET_REGISTRY[self.opt.train.dataset.train.name](device = self.device, **self.opt.train.dataset.train.args)
        self.train_dataloader = DATALOADER_REGISTRY[self.opt.train.dataset.train.dataloader.name](self.train_dataset, **self.opt.train.dataset.train.dataloader.args)
        print(f'load training dataset with {len(self.train_dataset)} samples')

        self.test_dataset = DATASET_REGISTRY[self.opt.train.dataset.test.name](device = self.device, **self.opt.train.dataset.test.args)
        self.test_dataloader = DATALOADER_REGISTRY[self.opt.train.dataset.test.dataloader.name](self.test_dataset, **self.opt.train.dataset.test.dataloader.args)
        print(f'load testing dataset with {len(self.test_dataset)} samples')
        
        print('-'*50)

    def train_prep(self):
        # init/load model, loss, & optimizer
        self.model = MODEL_REGISTRY[self.opt.model.name](device=self.device, **self.opt.model.args)
        self.loss_fn = LOSS_REGISTRY[self.opt.train.loss.name](**self.opt.train.loss.args)
        self.optimizer = OPTIMIZER_REGISTRY[self.opt.train.optimizer.name](**self.opt.train.optimizer.args, params=self.model.parameters())

        if self.opt.train.use_pretrain:
            self.model.load_state_dict(torch.load(Path(self.opt.path) / 'model.pth'))
            self.optimizer.load_state_dict(torch.load(Path(self.opt.path) / 'optimizer.pth'))

    def train_loop(self):
        self.start_time = time.time()

        for epoch in (pdar := tqdm(range(self.opt.train.optimizer.epochs))):
            self.logs['epoch'].append(epoch)
            self.logs['time'].append(time.strftime('%Y/%m/%d %H:%M:%S UTC', time.gmtime(time.time())))

            self._train_step(pdar)
            self._test_step(pdar)
            self._log_step()

    def _update_pdar(self, pdar, batch, batch_num, mode='training'):
        try:
            pdar.set_description(
                f'{mode} batch {batch}/{batch_num - 1} '\
                f'of epoch {self.logs["epoch"][-1]} | '\
                f'time {self.logs["time"][-1]} | '\
                f'train_loss {self.logs["train_loss"][-1]:.4f} '\
                f'test_loss {self.logs["test_loss"][-1]:.4f}')
        except:
            pdar.set_description(
                f'{mode} batch {batch}/{batch_num - 1} '\
                f'of epoch {self.logs["epoch"][-1]} | '\
                f'time {self.logs["time"][-1]} | '\
                f'train_loss {0:.4f} '\
                f'test_loss {0:.4f}')

    def _train_step(self, pdar):
        # put model in train mode
        self.model.train()
        train_loss = 0
        batch_num = len(self.train_dataloader)

        for batch, (x, y) in enumerate(self.train_dataloader):
            # progress bar update
            self._update_pdar(pdar, batch, batch_num, mode='training')

            # forward pass
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item()

            # loss backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # append loss
        self.logs['train_loss'].append(train_loss / len(self.train_dataloader))

    def _test_step(self, pdar):
        # put model in eval mode
        self.model.eval()

        # append new entry to logs
        test_loss = 0

        for key, metric_fn in self.metric_dict.items():
            self.logs[key].append(0)

        # turn on inference context manager
        with torch.inference_mode():
            batch_num = len(self.test_dataloader)

            for batch, (x, y) in enumerate(self.test_dataloader):
                # progress bar update
                self._update_pdar(pdar, batch, batch_num, mode='testing')
                    
                # forward pass
                y_pred = self.model(x)

                # loss calculation
                loss = self.loss_fn(y_pred, y)
                test_loss += loss.item()

                # metric calculation
                for key, metric_fn in self.metric_dict.items():
                    self.logs[key][-1] += metric_fn(y, y_pred).item()

        # averaging loss & metrics
        data_num = len(self.test_dataloader)
        self.logs['test_loss'].append(test_loss / data_num)

        for key, metric_fn in self.metric_dict.items():
            self.logs[key][-1] = self.logs[key][-1] / data_num

    def _log_step(self):
        # save logs
        pd.DataFrame(self.logs).to_csv(Path(self.opt.path) / 'logs.csv', index=False)

        # save model and optimizer if test loss is improved
        epoch = self.logs['epoch'][-1]

        if epoch == 0 or self.logs['test_loss'][-1] < min(self.logs['test_loss'][:-1]):
            torch.save(self.model.state_dict(), Path(self.opt.path) / 'model.pth')
            torch.save(self.optimizer.state_dict(), Path(self.opt.path) / 'optimizer.pth')


# --- benchmark scripts ---
@SCRIPT_REGISTRY.register()
class BenchmarkScript():
    def __init__(self, opt):
        self.opt = opt
        
        # device select
        if self.opt.device_select == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            if torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = self.opt.device_select

        print(f'running on {self.device}...')

        # init logs dict
        self.logs = {
            'dataset': [],
            'time': [],
        }

        # init metric dict
        self.metric_dict = {}

        for key, value in self.opt.benchmark.metric.items():
            self.metric_dict[key] = METRIC_REGISTRY[value.name](**value.args)
            self.logs[key] = []

    def load_data(self):
        self.dataset_dict = {}
        self.dataloader_dict = {}

        for key, value in self.opt.benchmark.dataset.items():
            self.dataset_dict[key] = DATASET_REGISTRY[value.name](device = self.device, **value.args)
            self.dataloader_dict[key] = DATALOADER_REGISTRY[value.dataloader.name](self.dataset_dict[key], **value.dataloader.args)
            print(f'load {key} dataset of {len(self.dataset_dict[key])} samples')

        print('-'*50)

    def load_model(self):
        self.model = MODEL_REGISTRY[self.opt.model.name](device=self.device, **self.opt.model.args)
        self.model.load_state_dict(torch.load(Path(self.opt.path) / 'model.pth'))

    def benchmark_loop(self):
        self.start_time = time.time()

        # for epoch in (pdar := tqdm(range(self.opt.train.optimizer.epochs))):
        for dataset_name in (pdar := tqdm(self.dataloader_dict.keys())):
            self.logs['dataset'].append(dataset_name)
            self.logs['time'].append(time.strftime('%Y/%m/%d %H:%M:%S UTC', time.gmtime(time.time())))

            self._benchmark_step(pdar, dataset_name)
            self._log_step()

    def _update_pdar(self, pdar, batch, batch_num, dataset_name):
        pdar.set_description(
            f'benchmarking batch {batch}/{batch_num - 1} '\
            f'of {dataset_name} dataset | '\
            f'time {self.logs["time"][-1]} | ')

    def _benchmark_step(self, pdar, dataset_name):
        # put model in eval mode
        self.model.eval()

        # append new entry to logs
        for key, metric_fn in self.metric_dict.items():
            self.logs[key].append(0)

        # turn on inference context manager
        with torch.inference_mode():
            batch_num = len(self.dataloader_dict[dataset_name])

            for batch, (x, y) in enumerate(self.dataloader_dict[dataset_name]):
                # progress bar update
                self._update_pdar(pdar, batch, batch_num, dataset_name)
                
                # forward pass
                y_pred = self.model(x)

                # metric calculation
                for key, metric_fn in self.metric_dict.items():
                    self.logs[key][-1] += metric_fn(y, y_pred).item()

        # averaging
        data_num = len(self.dataloader_dict[dataset_name])

        for key, metric_fn in self.metric_dict.items():
            self.logs[key][-1] = self.logs[key][-1] / data_num

    def _log_step(self):
        # save logs
        pd.DataFrame(self.logs).to_csv(Path(self.opt.path) / 'benchmarks.csv', index=False)


# --- sampling scripts for prediction example generation ---
@SCRIPT_REGISTRY.register()
class SamplingScript(BenchmarkScript):
    def __init__(self, opt):
        self.opt = opt
        
        # device select
        if self.opt.device_select == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            if torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = self.opt.device_select

        print(f'running on {self.device}...')

        # init logs dict
        self.logs = {
            'dataset': [],
            'time': [],
        }

        # init metric dict
        # p.s. the difference from BenchmarkScript is that opt is also inputted to the metric class
        self.metric_dict = {}

        for key, value in self.opt.sampling.metric.items():
            self.metric_dict[key] = METRIC_REGISTRY[value.name](opt=self.opt, **value.args)

    def load_data(self):
        self.dataset_dict = {}
        self.dataloader_dict = {}

        for key, value in self.opt.sampling.dataset.items():
            self.dataset_dict[key] = DATASET_REGISTRY[value.name](device = self.device, **value.args)
            self.dataloader_dict[key] = DATALOADER_REGISTRY[value.dataloader.name](self.dataset_dict[key], **value.dataloader.args)
            print(f'load {key} dataset of {len(self.dataset_dict[key])} samples')

        print('-'*50)

    def _benchmark_step(self, pdar, dataset_name):
        # put model in eval mode
        self.model.eval()

        # turn on inference context manager
        with torch.inference_mode():
            batch_num = len(self.dataloader_dict[dataset_name])
            batch_select = self.opt.sampling.dataset[dataset_name].batch_select

            for batch, (x, y) in enumerate(self.dataloader_dict[dataset_name]):
                # generate and store prediction samples of the selected batches only
                # p.s. storage code should be implemented in the metric class
                if batch in batch_select:
                    # progress bar update
                    self._update_pdar(pdar, batch, batch_num, dataset_name)
                    
                    # forward pass
                    y_pred = self.model(x)

                    # metric calculation
                    for key, metric_fn in self.metric_dict.items():
                        metric_fn(y_pred, y, dataset_name, batch)

                # break if all selected batches are processed to save time
                if batch == batch_select[-1]:
                    break

    def _log_step(self):
        # nothing to save
        pass