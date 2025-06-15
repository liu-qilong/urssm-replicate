from torch import nn

# customized base class
class BaseMetric(nn.Module):
    def forward(self):
        """Default method to invoke a forward pass"""
        pass

    def start_feed(self, script, name):
        """Method to start feeding the metric method. Typically called at the beginning of the testing/benchmark epoch
        
        Args:
            script: The traning/benchmark script object
        """
        self.script = script
        self.name = name

        # if self.eval() is called, prep for total metric gathering
        if not self.training:
            self.metric_total = 0.0
            self.metric_avg = 0.0
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
            self.metric_avg = self.metric_total / self.sample_total
            self.script.writer.add_scalar(f'metric/test/{self.name}', self.metric_avg, self.script.global_step)
