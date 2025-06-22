import torch
from torch import nn
import torch.distributed as dist

# customized base class
class BaseLoss(nn.Module):
    def forward(self):
        """Default method to invoke a forward pass"""
        pass

    def start_feed(self, script, name, rank=None):
        """Method to start feeding the loss method. Typically called at the beginning of the testing/benchmark epoch
        
        Args:
            script: The traning/benchmark script object
            name: The name of the loss method
            rank: The rank of the process (if using distributed training)
        """
        self.script = script
        self.name = name
        self.rank = rank

        # if self.eval() is called, prep for total loss gathering
        if not self.training:
            self.loss_total = 0.0
            self.loss_avg = 0.0
            self.sample_total = 0

    def feed(self, infer, data):
        """Method to feed the loss method with inference results and data. Typically called during the testing/benchmark epoch
        """
        loss_val = self(infer, data)
        
        # if self.eval() is called, gather total loss
        if not self.training:
            self.loss_total += loss_val
            self.sample_total += len(data)
        
        return loss_val

    def end_feed(self):
        """Method to end feeding the loss method. Typically called at the end of the testing/benchmark epoch
        """
        # if self.eval() is called, log avg loss
        if not self.training:
            if self.rank is None:
                # if not using distributed training, simply log the average loss
                self.loss_avg = self.loss_total / self.sample_total
                self.script.writer.add_scalar(f'loss/test/{self.name}', self.loss_avg, self.script.global_step)

            else:
                # if using distributed training, gather the total loss and sample count across all ranks
                # only log in rank 0
                self.sample_total = torch.tensor(self.sample_total).to(device=self.rank)
                dist.all_reduce(self.loss_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(self.sample_total, op=dist.ReduceOp.SUM)

                if self.rank == 0:
                    self.loss_avg = self.loss_total / self.sample_total
                    self.script.writer.add_scalar(f'loss/test/{self.name}', self.loss_avg, self.script.global_step)
