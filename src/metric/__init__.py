import torch
from torch import nn
import torch.distributed as dist

# customized base class
class BaseMetric(nn.Module):
    def forward(self):
        """Default method to invoke a forward pass"""
        pass

    def start_feed(self, script, name, rank=None):
        """Method to start feeding the metric method. Typically called at the beginning of the testing/benchmark epoch
        
        Args:
            script: The traning/benchmark script object
            name: The name of the metric method
            rank: The rank of the process (if using distributed training)
        """
        self.script = script
        self.name = name
        self.rank = rank

        # if self.eval() is called, prep for total metric gathering
        if not self.training:
            self.metric_total = 0.0
            self.sample_total = 0

    def feed(self, infer, data):
        """Method to feed the metric method with inference results and data. Typically called during the testing/benchmark epoch
        """
        metric_val = self(infer, data)
        
        # if self.eval() is called, gather total metric
        if not self.training:
            self.metric_total += metric_val
            self.sample_total += len(data)
        
        return metric_val

    def end_feed(self):
        """Method to end feeding the metric method. Typically called at the end of the testing/benchmark epoch. Metric values of the whole test epoch could be averaged and logged in this method.
        
        Noted that `self.metric_avg` will be used to determin whether the current model is the best so far and thus be checkpointed
        """
        # if self.eval() is called, log avg metric
        if not self.training:
            if self.rank is None:
                # if not using distributed training, simply log the average loss
                self.script.writer.add_scalar(f'metric/test/{self.name}', self.metric_total / self.sample_total, self.script.global_step)

            else:
                # if using distributed training, gather the total loss and sample count across all ranks
                self.sample_total = torch.tensor(
                    self.sample_total,
                    device=self.device,
                )
                dist.all_reduce(self.metric_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(self.sample_total, op=dist.ReduceOp.SUM)

                if self.rank == 0:
                    # only log in rank 0
                    self.script.writer.add_scalar(f'metric/test/{self.name}', self.metric_total / self.sample_total, self.script.global_step)

