import os
from tqdm.auto import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from src.infra.registry import DATASET_REGISTRY, DATALOADER_REGISTRY, NETWORK_REGISTRY, MODULE_REGISTRY, LOSS_REGISTRY, METRIC_REGISTRY, OPTIMIZER_REGISTRY, SCRIPT_REGISTRY
from src.utils.tensor import to_device


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
        self.writer = SummaryWriter(os.path.join(self.opt.path, 'logs'))
        os.makedirs(os.path.join(self.opt.path, 'checkpoint'), exist_ok=True)

    def load_data(self):
        self.train_dataset = DATASET_REGISTRY[self.opt.train.dataset.train.name](**self.opt.train.dataset.train.args)
        self.train_dataloader = DATALOADER_REGISTRY[self.opt.train.dataset.train.dataloader.name](self.train_dataset, **self.opt.train.dataset.train.dataloader.args)
        print(f'load training dataset with {len(self.train_dataset)} samples')

        self.test_dataset = DATASET_REGISTRY[self.opt.train.dataset.test.name](**self.opt.train.dataset.test.args)
        self.test_dataloader = DATALOADER_REGISTRY[self.opt.train.dataset.test.dataloader.name](self.test_dataset, **self.opt.train.dataset.test.dataloader.args)
        print(f'load testing dataset with {len(self.test_dataset)} samples')

    def train_prep(self):
        # init/load model
        self.network = NETWORK_REGISTRY[self.opt.network.name](self.opt).to(self.device)
        print(f'initialized model {self.opt.network.name}')

        if 'load_from' in self.opt.network:
            pth_path = self.opt.network.load_from
            self.network.load_state_dict(torch.load(pth_path, map_location=self.device))
            print(f'loaded model weights from {pth_path}')

        # init optimizer
        self.optimizer = OPTIMIZER_REGISTRY[self.opt.train.optimizer.name](**self.opt.train.optimizer.args, params=self.network.parameters())

        if 'load_from' in self.opt.train.optimizer:
            pth_path = self.opt.train.optimizer.load_from
            self.optimizer.load_state_dict(torch.load(pth_path, map_location=self.device))            
            print(f'loaded optimizer weights from {pth_path}')

        # init loss dict
        self.loss_dict = {}

        for name, loss in self.opt.train.loss.items():
            self.loss_dict[name] = {
                'fn': LOSS_REGISTRY[loss['name']](**loss['args']).to(self.device),
                'weight': loss['weight'],
            }
        
        # init metric dict
        self.metric_dict = {}

        for name, metric in self.opt.train.metric.items():
            self.metric_dict[name] = METRIC_REGISTRY[metric['name']](**metric['args']).to(self.device)

    def train_loop(self):
        epochs = self.opt.train.optimizer.epochs
        self.global_step = 0
        self.best_bench = torch.inf
        
        for epoch in range(epochs):
            for batch, data in (pdar := tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))):
                pdar.set_description(f"epoch {epoch}/{epochs-1}")
                self.global_step += 1

                self.network.train()
                data = to_device(data, self.device)
                self._train_step(data)

                if self.global_step != 0 and self.global_step % self.opt.train.test_interval == 0:
                    self._test_step(pdar)

                if self.global_step != 0 and self.global_step % self.opt.train.checkpoint_interval == 0:
                    self._checkpoint_step()

    def _train_step(self, data):
        # forward pass & loss calculation
        infer = self.network(data)
        loss_total = 0.0

        for name, loss in self.loss_dict.items():
            loss_val = loss['fn'](infer, data) * loss['weight']
            loss_total += loss_val
            self.writer.add_scalar(f'loss/train/{name}', loss_val.item(), self.global_step)

        # loss backward
        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

    def _test_step(self, pdar):
        batches = len(self.test_dataloader)
        loss_val = { name: 0.0 for name in self.loss_dict.keys() }
        metric_val = { name: 0.0 for name in self.metric_dict.keys() }

        # turn on inference context manager & run the testing
        self.network.eval()
        with torch.inference_mode():
            for batch, data in enumerate(self.test_dataloader):
                pdar.set_description(f"testing batch {batch}/{batches-1}")
                data = to_device(data, self.device)
                
                # forward pass
                infer = self.network(data)

                # loss calculation
                for name, loss in self.loss_dict.items():
                    loss_val[name] += loss['fn'](infer, data) * loss['weight']

                # metric calculation
                for name, metric in self.metric_dict.items():
                    metric_val[name] += metric(infer, data)

        # logging
        for name, val in loss_val.items():
            self.writer.add_scalar(f'loss/test/{name}', val / batches, self.global_step)

        for name, val in metric_val.items():
            self.writer.add_scalar(f'metric/test/{name}', val / batches, self.global_step)

        # save current model if it's the best so far
        def save_best_model():
            torch.save(self.network.state_dict(), os.path.join(self.opt.path, 'checkpoint', f'model-{self.global_step}-best.pth'))
            torch.save(self.optimizer.state_dict(), os.path.join(self.opt.path, 'checkpoint', f'optimizer-{self.global_step}-best.pth'))
        
        bench_name = self.opt.train.save_best

        if bench_name in self.loss_dict and self.best_bench > loss_val[bench_name]:
            self.best_bench = loss_val[bench_name]
            save_best_model()

        if bench_name in self.metric_dict and self.best_bench > metric_val[bench_name]:
            self.best_bench = metric_val[bench_name]
            save_best_model()

    def _checkpoint_step(self):
        # save model
        torch.save(self.network.state_dict(), os.path.join(self.opt.path, 'checkpoint', f'model-{self.global_step}.pth'))

        # save optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(self.opt.path, 'checkpoint', f'optimizer-{self.global_step}.pth'))


# --- bench scripts ---
# @SCRIPT_REGISTRY.register()
# class BenchScript():
#     def __init__(self, opt):
#         self.opt = opt
        
#         # device select
#         if self.opt.device_select == 'auto':
#             if torch.cuda.is_available():
#                 self.device = 'cuda'
#             if torch.backends.mps.is_available():
#                 self.device = 'mps'
#             else:
#                 self.device = 'cpu'
#         else:
#             self.device = self.opt.device_select

#         print(f'running on {self.device}...')

#         # init metric dict
#         self.metric_dict = {}

#         for key, value in self.opt.benchmark.metric.items():
#             self.metric_dict[key] = METRIC_REGISTRY[value.name](**value.args)

#     def load_data(self):
#         self.dataset_dict = {}
#         self.dataloader_dict = {}

#         for key, value in self.opt.benchmark.dataset.items():
#             self.dataset_dict[key] = DATASET_REGISTRY[value.name](device = self.device, **value.args)
#             self.dataloader_dict[key] = DATALOADER_REGISTRY[value.dataloader.name](self.dataset_dict[key], **value.dataloader.args)
#             print(f'load {key} dataset of {len(self.dataset_dict[key])} samples')

#     def load_model(self):
#         self.network = MODULE_REGISTRY[self.opt.network.name](**self.opt.network.args)
#         self.network.to(self.device)
#         self.network.load_state_dict(torch.load(Path(self.opt.path) / 'model.pth'))

#     def benchmark_loop(self):
#         self.start_time = time.time()

#         # for epoch in (pdar := tqdm(range(self.opt.train.optimizer.epochs))):
#         for dataset_name in (pdar := tqdm(self.dataloader_dict.keys())):
#             self.logs['dataset'].append(dataset_name)
#             self.logs['time'].append(time.strftime('%Y/%m/%d %H:%M:%S UTC', time.gmtime(time.time())))

#             self._benchmark_step(pdar, dataset_name)
#             self._log_step()

#     def _update_pdar(self, pdar, batch, batch_num, dataset_name):
#         pdar.set_description(
#             f'benchmarking batch {batch}/{batch_num - 1} '\
#             f'of {dataset_name} dataset | '\
#             f'time {self.logs["time"][-1]} | ')

#     def _benchmark_step(self, pdar, dataset_name):
#         # put model in eval mode
#         self.network.eval()

#         # append new entry to logs
#         for key, metric_fn in self.metric_dict.items():
#             self.logs[key].append(0)

#         # turn on inference context manager
#         with torch.inference_mode():
#             batch_num = len(self.dataloader_dict[dataset_name])

#             for batch, (x, y) in enumerate(self.dataloader_dict[dataset_name]):
#                 # progress bar update
#                 self._update_pdar(pdar, batch, batch_num, dataset_name)
                
#                 # forward pass
#                 y_pred = self.network(x)

#                 # metric calculation
#                 for key, metric_fn in self.metric_dict.items():
#                     self.logs[key][-1] += metric_fn(y, y_pred).item()

#         # averaging
#         data_num = len(self.dataloader_dict[dataset_name])

#         for key, metric_fn in self.metric_dict.items():
#             self.logs[key][-1] = self.logs[key][-1] / data_num

#     def _log_step(self):
#         # save logs
#         pd.DataFrame(self.logs).to_csv(Path(self.opt.path) / 'benchmarks.csv', index=False)