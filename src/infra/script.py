import time
import shutil
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from src.infra.registry import DATASET_REGISTRY, DATALOADER_REGISTRY, NETWORK_REGISTRY, MODULE_REGISTRY, LOSS_REGISTRY, METRIC_REGISTRY, OPTIMIZER_REGISTRY, SCRIPT_REGISTRY
from src.utils.tensor import to_device
from src.utils.misc import format_time


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

    def run(self):
        self.setup_dir()
        self.load_data()
        self.train_prep()
        self.train_loop()

    def setup_dir(self):
        Path(self.opt.path).mkdir()
        print(f'created experiment folder {self.opt.path}')

        shutil.copy(self.opt.config, Path(self.opt.path) / 'config.yaml')
        print(f"copied configurations to {Path(self.opt.path) / 'config.yaml'}")

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

    def print_training_progress(self, epoch, epoch_num, batch, batch_num, step, step_total, start_time):
        progress_rate = step / step_total
        elapsed_time = time.time() - start_time
        step_throughput = (step + 1) / elapsed_time
        eta_time = elapsed_time / progress_rate - elapsed_time if progress_rate > 0 else 0
        print(f"training | epoch {epoch}/{epoch_num} batch {batch}/{batch_num - 1} ({progress_rate * 100:.2f}%) | throughput {step_throughput:.2f} batch/s | elapsed {format_time(elapsed_time)} | eta {format_time(eta_time)}")

    def print_testing_progress(self, batch, batch_num, start_time):
        progress_rate = batch / batch_num
        elapsed_time = time.time() - start_time
        batch_throughput = (batch + 1) / elapsed_time
        eta_time = elapsed_time / progress_rate - elapsed_time if progress_rate > 0 else 0
        print(f"testing | batch {batch}/{batch_num - 1} ({progress_rate * 100:.2f}%) | throughput: {batch_throughput:.2f} batch/s | elapsed: {format_time(elapsed_time)} | eta: {format_time(eta_time)}")

    def train_loop(self):
        epochs = self.opt.train.optimizer.epochs
        start_time = time.time()
        self.global_step = 0
        self.total_step = epochs * len(self.train_dataloader)
        batch_num = len(self.train_dataloader)
        self.best_bench = torch.inf

        for epoch in range(epochs):
            for batch, data in enumerate(self.train_dataloader):
                self.global_step += 1
                self.print_training_progress(epoch, epochs, batch, batch_num, self.global_step, self.total_step, start_time)

                data = to_device(data, self.device)
                self._train_step(data)

                if self.global_step % self.opt.train.test_interval == 0:
                    self._test_step()

                if self.global_step % self.opt.train.checkpoint_interval == 0:
                    self._checkpoint_step()

        # final test & checkpoint
        print('running final test & checkpoint...')
        self._test_step()
        self._checkpoint_step()
        self.writer.flush()

    def _train_step(self, data):
        self.network.train()

        # forward pass & loss calculation
        infer = self.network(data)
        loss_total = 0.0

        for name, loss in self.loss_dict.items():
            loss.train()
            loss_val = loss.feed(infer, data)
            loss_total += loss_val * self.loss_weight[name]
            self.writer.add_scalar(f'loss/train/{name}', loss_val.item() / self.train_dataloader.batch_size, self.global_step)

        # loss backward
        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

    def _test_step(self):
        start_time = time.time()
        batch_num = len(self.test_dataloader)

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
                self.print_testing_progress(batch, batch_num, start_time)
                
                # forward pass
                data = to_device(data, self.device)
                infer = self.network(data)

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


@SCRIPT_REGISTRY.register()
class DDPTrainScript(TrainScript):
    def __init__(self, opt, rank, world_size):
        self.opt = opt
        self.rank = rank
        self.world_size = world_size
        
        # device select
        self.device = self.rank
        print(f'[rank {self.rank}] launched')

    def run(self):
        if self.rank == 0:
            self.setup_dir()

        self.load_data()
        self.train_prep()
        self.train_loop()

    def load_data(self):
        self.train_dataset = DATASET_REGISTRY[self.opt.train.dataset.train.name](**self.opt.train.dataset.train.args)
        self.train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )
        self.train_dataloader = DATALOADER_REGISTRY[self.opt.train.dataset.train.dataloader.name](
            self.train_dataset,
            sampler=self.train_sampler,
            shuffle=False,
            **self.opt.train.dataset.train.dataloader.args,
        )
        print(f'[rank {self.rank}] will sample {len(self.train_sampler)} out of {len(self.train_dataset)} samples for training')

        self.test_dataset = DATASET_REGISTRY[self.opt.train.dataset.test.name](**self.opt.train.dataset.test.args)
        self.test_sampler = DistributedSampler(
            self.test_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )
        self.test_dataloader = DATALOADER_REGISTRY[self.opt.train.dataset.test.dataloader.name](
            self.test_dataset,
            sampler=self.test_sampler,
            shuffle=False,
            **self.opt.train.dataset.test.dataloader.args,
        )
        print(f'[rank {self.rank}] will sample {len(self.test_sampler)} out of {len(self.test_dataset)} samples for testing')
        
    def train_prep(self):
        # init/load model
        self.network = NETWORK_REGISTRY[self.opt.network.name](self.opt).to(self.device)
        print(f'[rank {self.rank}] initialized model {self.opt.network.name}')

        if 'load_from' in self.opt.network:
            pth_path = self.opt.network.load_from
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
            self.network.load_state_dict(torch.load(pth_path, map_location=map_location))
            print(f'[rank {self.rank}] loaded model weights from {pth_path}')

        self.ddp_network = DDP(self.network, device_ids=[self.rank])

        # init optimizer
        self.opt.train.optimizer.args.lr *= self.world_size  # scale learning rate by world size to keep the same effective learning rate per sample
        self.optimizer = OPTIMIZER_REGISTRY[self.opt.train.optimizer.name](**self.opt.train.optimizer.args, params=self.ddp_network.parameters())

        if 'load_from' in self.opt.train.optimizer:
            pth_path = self.opt.train.optimizer.load_from
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
            self.optimizer.load_state_dict(torch.load(pth_path, map_location=map_location))
            print(f'[rank {self.rank}] loaded optimizer weights from {pth_path}')

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
        epochs = self.opt.train.optimizer.epochs
        start_time = time.time()
        self.global_step = 0
        self.total_step = epochs * len(self.train_dataloader)
        batch_num = len(self.train_dataloader)
        self.best_bench = torch.inf

        for epoch in range(epochs):
            self.train_sampler.set_epoch(epoch)

            for batch, data in enumerate(self.train_dataloader):
                self.global_step += 1
                self.print_training_progress(epoch, epochs, batch, batch_num, self.global_step, self.total_step, start_time)

                self._train_step(data)

                if self.global_step % self.opt.train.test_interval == 0:
                    self._test_step()

                if self.global_step % self.opt.train.checkpoint_interval == 0:
                    self._checkpoint_step()

        # final test & checkpoint
        print('running final test & checkpoint...')
        self._test_step()
        self._checkpoint_step()
        
        if self.rank == 0:
            self.writer.flush()

    def _train_step(self, data):
        data = to_device(data, self.device)
        self.ddp_network.train()

        # forward pass & loss calculation
        infer = self.ddp_network(data)
        loss_total = 0.0

        for name, loss in self.loss_dict.items():
            loss.train()
            loss_val = loss.feed(infer, data)
            loss_total += loss_val * self.loss_weight[name]
            
            if self.rank == 0:
                # only log training loss of rank 0
                # p.s. gather losses from all ranks w/ all reduce is possible
                # but it unnecessarily will introduce communication overhead
                self.writer.add_scalar(f'loss/train/{name}', loss_val.item() / self.train_dataloader.batch_size, self.global_step)

        # loss backward
        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

    def _test_step(self):
        start_time = time.time()
        batch_num = len(self.test_dataloader)

        # turn on eval mode and start feed
        self.ddp_network.eval()

        for name, loss in self.loss_dict.items():
            loss.eval()
            loss.start_feed(self, name, self.rank)

        for name, metric in self.metric_dict.items():
            metric.eval()
            metric.start_feed(self, name, self.rank)

        # turn on inference context manager & run the testing
        with torch.inference_mode():
            for batch, data in enumerate(self.test_dataloader):
                self.print_testing_progress(batch, batch_num, start_time)
                
                # forward pass
                data = to_device(data, self.device)
                infer = self.ddp_network(data)

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
        # p.s. only in rank 0
        if self.rank == 0:
            def save_best_model():
                torch.save(self.ddp_network.state_dict(), Path(self.opt.path) / 'checkpoint' / f'model-best.pth')
                torch.save(self.optimizer.state_dict(), Path(self.opt.path) / 'checkpoint' / f'optimizer-best.pth')
            
            bench_name = self.opt.train.save_best

            if bench_name in self.loss_dict and self.best_bench > self.loss_dict[bench_name].loss_avg:
                self.best_bench = self.loss_dict[bench_name].loss_avg
                save_best_model()

            if bench_name in self.metric_dict and self.best_bench > self.metric_dict[bench_name].metric_avg:
                self.best_bench = self.metric_dict[bench_name].metric_avg
                save_best_model()

    def _checkpoint_step(self):
        if self.rank == 0:
            # save model
            torch.save(self.ddp_network.state_dict(), Path(self.opt.path) / 'checkpoint' / f'model-{self.global_step}.pth')

            # save optimizer state
            torch.save(self.optimizer.state_dict(), Path(self.opt.path) / 'checkpoint' / f'optimizer-{self.global_step}.pth')


# --- benchmark scripts ---
@SCRIPT_REGISTRY.register()
class BenchScript:
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

    def run(self):
        self.writer = SummaryWriter(Path(self.opt.path) / 'bench')
        self.load_data()
        self.bench_prep()
        self.bench_loop()

    def load_data(self):
        self.test_dataset = DATASET_REGISTRY[self.opt.benchmark.dataset.name](**self.opt.benchmark.dataset.args)
        self.test_dataloader = DATALOADER_REGISTRY[self.opt.benchmark.dataset.dataloader.name](self.test_dataset, **self.opt.benchmark.dataset.dataloader.args)
        print(f'load testing dataset with {len(self.test_dataset)} samples')

    def bench_prep(self):
        # init model
        self.network = NETWORK_REGISTRY[self.opt.network.name](self.opt).to(self.device)
        print(f'initialized model {self.opt.network.name}')

        # load model weights
        if 'network_weight' in self.opt.benchmark:
            pth_path = self.opt.benchmark.network_weight

        else:
            pth_path = Path(self.opt.path) / 'checkpoint' / 'model-best.pth'

        state_dict = torch.load(pth_path, map_location=self.device)
        if any(k.startswith('module.') for k in state_dict.keys()):
            # handle DDP 'module.' prefix
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

        self.network.load_state_dict(state_dict)
        print(f'loaded model weights from {pth_path}')

        # init loss dict
        self.loss_dict = {}
        self.loss_weight = {}

        for name, loss in self.opt.benchmark.loss.items():
            self.loss_dict[name] = LOSS_REGISTRY[loss['name']](**loss['args']).to(self.device)
            self.loss_weight[name] = loss['weight']
        
        # init metric dict
        self.metric_dict = {}

        for name, metric in self.opt.benchmark.metric.items():
            self.metric_dict[name] = METRIC_REGISTRY[metric['name']](**metric['args']).to(self.device)

    def print_benchmarking_progress(self, batch, batch_num, start_time):
        progress_rate = batch / batch_num
        elapsed_time = time.time() - start_time
        batch_throughput = (batch + 1) / elapsed_time
        eta_time = elapsed_time / progress_rate - elapsed_time if progress_rate > 0 else 0
        print(f"benchmarking | batch {batch}/{batch_num - 1} ({progress_rate * 100:.2f}%) | throughput: {batch_throughput:.2f} batch/s | elapsed: {format_time(elapsed_time)} | eta: {format_time(eta_time)}")

    def bench_loop(self):
        self.global_step = 0
        batch_num = len(self.test_dataloader)
        start_time = time.time()

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
                self.print_benchmarking_progress(batch, batch_num, start_time)
                
                # forward pass
                data = to_device(data, self.device)
                infer = self.network(data)

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

        self.writer.flush()