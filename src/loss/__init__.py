from torch import nn

# customized base class
class BaseLoss(nn.Module):
    def forward(self):
        """Default method to invoke a forward pass"""
        pass

    def start_feed(self, script, name):
        """Method to start feeding the loss method. Typically called at the beginning of the testing/benchmark epoch
        
        Args:
            script: The traning/benchmark script object
        """
        self.script = script
        self.name = name

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
            self.loss_avg = self.loss_total / self.sample_total
            self.script.writer.add_scalar(f'loss/test/{self.name}', self.loss_avg, self.script.global_step)
