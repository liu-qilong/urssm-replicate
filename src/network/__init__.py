import torch.nn as nn

# customized base class
class BaseNetwork(nn.Module):
    def forward(self):
        """Default method to invoke a forward pass"""
        pass

    def start_feed(self, script):
        """Method to start feeding the network. Typically called at the beginning of the traning/benchmark epoch
        
        Args:
            script: The traning/benchmark script object
        """
        self.script = script

    def feed(self, infer, data):
        """Method to feed the network with inference results and data. Typically called during the traning/benchmark epoch
        """
        pass

    def end_feed(self):
        """Method to end feeding the network. Typically called at the end of the traning/benchmark epoch"""
        pass
