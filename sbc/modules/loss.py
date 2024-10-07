import torch

from mace.tools import TensorDict
from mace.tools.torch_geometric import Batch
from mace.tools.scatter import scatter_sum


class ClassificationLoss(torch.nn.Module):
    def __init__(self, weight=None, label_smoothing=0.0, energy_weight=0.0):
        super().__init__()
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.get_default_dtype())
        self.ce_loss = torch.nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                weight=weight,
                )
        self.register_buffer(
            'energy_weight', torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, output: TensorDict) -> torch.Tensor:
        nl = output['node_logits']
        node_delta = output['node_delta']
        node_inverse = output['node_inverse']

        cross_entropy = self.ce_loss(output['logits'], ref['phase'].to(torch.long))
        RMSE = torch.sqrt(torch.mean(torch.square(node_delta)))
        return cross_entropy + self.energy_weight * RMSE

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
        )
