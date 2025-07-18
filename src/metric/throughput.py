import time
import torch
import torch.distributed as dist

from src.metric import BaseMetric
from src.infra.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
class Throughput(BaseMetric):
    """The principle of throughput measurement is logging the time taken for the whole testing epoch, and then dividing the total number of samples by the time taken. Since the time for inference and the time for calculating other metrics can't be distinguished, this metric is meant to used **alone**. You can:
    
    1. First benchmark other metrics in one `bench.py` run
    2. Comment out other metrics and set `opt.benchmark.loss: {}`.
    3. Add this metric for another `bench.py` run. Throughput results will be added and won't affect the results of other metrics.
    """
    def forward(self, infer, data):
        """Do nothing
        """
        return torch.tensor(0.0).to(device=self.script.device)
    
    def start_feed(self, script, name, rank=None):
        """Log start time
        
        Args:
            script: The traning/benchmark script object
            name: The name of the metric method
            rank: The rank of the process (if using distributed training)
        """
        super().start_feed(script, name, rank)
        assert len(self.script.opt.benchmark.loss) + len(self.script.opt.benchmark.metric) == 1, "Throughput metric should be used alone, please comment out other metrics in the config file. You can: 1. First benchmark other metrics in one `bench.py` run 2. Comment out other metrics and set `opt.benchmark.loss: {}`. 3. Add this metric for another `bench.py` run. Throughput results will be added and won't affect the results of other metrics."
        self.start_time = time.time()

    def end_feed(self):
        """Method to end feeding the metric method. Typically called at the end of the testing/benchmark epoch. Metric values of the whole test epoch could be averaged and logged in this method.
        
        Noted that `self.metric_avg` will be used to determin whether the current model is the best so far and thus be checkpointed
        """
        # if self.eval() is called, log avg metric
        if not self.training:
            elapsed_time = time.time() - self.start_time

            if self.rank is None:
                # if not using distributed training, simply log the average loss
                self.metric_avg = self.sample_total / elapsed_time
                self.script.writer.add_scalar(f'metric/test/{self.name}', self.metric_avg, self.script.global_step)

            else:
                # if using distributed training, gather the total loss and sample count across all ranks
                self.sample_total = torch.tensor(self.sample_total).to(device=self.rank)
                dist.all_reduce(self.metric_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(self.sample_total, op=dist.ReduceOp.SUM)

                if self.rank == 0:
                    # only log in rank 0
                    self.metric_avg = elapsed_time / self.sample_total
                    self.script.writer.add_scalar(f'metric/test/{self.name}', self.metric_avg, self.script.global_step)
