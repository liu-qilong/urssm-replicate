from pathlib import Path
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
        self.writer = SummaryWriter(Path(self.opt.path) / 'log')
        (Path(self.opt.path) / 'checkpoint').mkdir()

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
        self.loss_weight = {}

        for name, loss in self.opt.train.loss.items():
            self.loss_dict[name] = LOSS_REGISTRY[loss['name']](**loss['args']).to(self.device)
            self.loss_weight[name] = loss['weight']
        
        # init metric dict
        self.metric_dict = {}

        for name, metric in self.opt.train.metric.items():
            self.metric_dict[name] = METRIC_REGISTRY[metric['name']](**metric['args']).to(self.device)

    def train_loop(self):
        print('-' * 100)
        epochs = self.opt.train.optimizer.epochs
        self.global_step = 0
        self.best_bench = torch.inf
        self.network.start_feed(self)

        for epoch in range(epochs):
            for batch, data in (pdar := tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), dynamic_ncols=True)):
                pdar.set_description(f"epoch {epoch}/{epochs-1}")
                self.global_step += 1

                data = to_device(data, self.device)
                self._train_step(data)

                if self.global_step != 0 and self.global_step % self.opt.train.test_interval == 0:
                    self._test_step(pdar)

                if self.global_step != 0 and self.global_step % self.opt.train.checkpoint_interval == 0:
                    self._checkpoint_step()

        self.network.end_feed()

        # final test & checkpoint
        self._test_step()
        self._checkpoint_step()
        self.writer.flush()

    def _train_step(self, data):
        self.network.train()

        # forward pass & loss calculation
        infer = self.network.feed(data)
        loss_total = 0.0

        for name, loss in self.loss_dict.items():
            loss.train()
            loss_val = loss.feed(infer, data)
            loss_total += loss_val * self.loss_weight[name]
            self.writer.add_scalar(f'loss/train/{name}', loss_val.item(), self.global_step)

        # loss backward
        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

    def _test_step(self, pdar=None):
        # turn on eval mode and start feed
        self.network.eval()

        for name, loss in self.loss_dict.items():
            loss.eval()
            loss.start_feed(self, name)

        for name, metric in self.metric_dict.items():
            metric.eval()
            metric.start_feed(self, name)

        # turn on inference context manager & run the testing
        with torch.inference_mode():
            for batch, data in enumerate(self.test_dataloader):
                if batch > 10:
                    break

                if pdar is not None:
                    pdar.set_description(f"testing batch {batch}/{len(self.test_dataloader) - 1}")
                
                data = to_device(data, self.device)
                
                # forward pass
                infer = self.network.feed(data)

                # loss calculation
                for name, loss in self.loss_dict.items():
                    loss.feed(infer, data)

                # metric calculation
                for name, metric in self.metric_dict.items():
                    metric.feed(infer, data)

        # end feed
        for name, loss in self.loss_dict.items():
            loss.end_feed()

        for name, metric in self.metric_dict.items():
            metric.end_feed()

        # save current model if it's the best so far
        def save_best_model():
            torch.save(self.network.state_dict(), Path(self.opt.path) / 'checkpoint' / f'model-best.pth')
            torch.save(self.optimizer.state_dict(), Path(self.opt.path) / 'checkpoint' / f'optimizer-best.pth')
        
        bench_name = self.opt.train.save_best

        if bench_name in self.loss_dict and self.best_bench > self.loss_dict[bench_name].loss_avg:
            self.best_bench = self.loss_dict[bench_name].loss_avg
            save_best_model()

        if bench_name in self.metric_dict and self.best_bench > self.metric_dict[bench_name].metric_avg:
            self.best_bench = self.metric_dict[bench_name].metric_avg
            save_best_model()

    def _checkpoint_step(self):
        # save model
        torch.save(self.network.state_dict(), Path(self.opt.path) / 'checkpoint' / f'model-{self.global_step}.pth')

        # save optimizer state
        torch.save(self.optimizer.state_dict(), Path(self.opt.path) / 'checkpoint' / f'optimizer-{self.global_step}.pth')
