from typing import Optional, Callable

import numpy as np
import torch

from e3nn import o3, nn
from e3nn.util.jit import compile_mode

from mace.tools import TensorDict
from mace.tools.scatter import scatter_sum, scatter_mean


@compile_mode("script")
class InvariantClassifierReadoutBlock(torch.nn.Module):
    def __init__(
            self, irreps_in: o3.Irreps, layers: list[int]):
        super().__init__()
        assert len(layers) > 0
        self.layers = torch.nn.ModuleList()

        for size in layers:
            irreps = o3.Irreps('{}x0e'.format(size))
            self.layers.append(o3.Linear(irreps_in=irreps_in, irreps_out=irreps))
            self.layers.append(nn.Activation(irreps_in=irreps, acts=[torch.nn.functional.silu]))
            irreps_in = irreps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        for layer in self.layers:
            x = layer(x)
        return x


@compile_mode('script')
class Classifier(torch.nn.Module):
    def __init__(
            self,
            phases: list[str],
            num_interactions: int,
            hidden_irreps: o3.Irreps,
            readout_layers: list[int],
            mixing_layer: Optional[int],
            ):
        super().__init__()
        self.phases = phases
        self.num_interactions = num_interactions
        self.readout_layers = readout_layers
        self.mixing_layer = mixing_layer

        self.l_max = max([ir[1][0] for ir in hidden_irreps])
        self.num_features = hidden_irreps[0].dim

        self.readouts = torch.nn.ModuleList()
        for irreps in [hidden_irreps] * (num_interactions - 1) + [hidden_irreps[:1]]:
            block = InvariantClassifierReadoutBlock(
                irreps,
                readout_layers,
                )
            self.readouts.append(block)

        self.mix = torch.nn.ModuleList()
        if mixing_layer is not None:
            self.mix.append(torch.nn.Linear(
                in_features=(num_interactions * readout_layers[-1]),
                out_features=mixing_layer,
                ))
            self.mix.append(torch.nn.SiLU())
            self.mix.append(torch.nn.Linear(in_features=mixing_layer, out_features=len(phases), bias=False))
        else:
            self.mix.append(torch.nn.Linear(in_features=num_interactions * readout_layers[-1], out_features=len(phases), bias=True))
        self.register_buffer('scale', torch.nn.Parameter(torch.tensor(1.0, dtype=torch.get_default_dtype())))


class SimpleClassifier(Classifier):

    def forward(self, **data: torch.Tensor) -> TensorDict:
        node_feats = data['node_feats']

        features = []  # from mace/modules/utils.py#L171
        for i in range(self.num_interactions - 1):
            features.append(
                node_feats[
                    :,
                    i
                    * (self.l_max + 1) ** 2  # 1, 1 + 3, 1 + 3 + 5, ...
                    * self.num_features : ((i + 1) * (self.l_max + 1) ** 2)
                    * self.num_features,
                ]
            )
        features.append(node_feats[:, -self.num_features:])

        y = torch.cat([read(f) for read, f in zip(self.readouts, features)], dim=-1)
        for layer in self.mix:
            y = layer(y)

        # normalization breaks training!?
        #node_logits = y - torch.logsumexp(y, dim=-1, keepdim=True)
        node_logits = y
        probabilities = torch.nn.functional.softmax(node_logits, dim=-1)
        node_deltas = (-1.0) * node_logits
        node_delta = torch.sum(probabilities * node_deltas, dim=-1)
        logits = scatter_sum(
                src=node_logits,
                index=data['batch'],
                dim=0,
                dim_size=data["ptr"].numel() - 1,
                )
        deltas = scatter_sum(
                src=node_deltas,
                index=data['batch'],
                dim=0,
                dim_size=data["ptr"].numel() - 1,
                )
        delta = scatter_sum(
                src=node_delta,
                index=data['batch'],
                dim=0,
                dim_size=data["ptr"].numel() - 1,
                )

        return {
                'node_logits': node_logits,
                'node_deltas': node_deltas,
                'node_delta': node_delta,
                'logits': logits,
                'deltas': deltas,
                'delta': delta,
                }


class EnergyBasedClassifier(Classifier):

    def forward(self, **data: torch.Tensor) -> TensorDict:
        node_feats = data['node_feats']

        features = []  # from mace/modules/utils.py#L171
        for i in range(self.num_interactions - 1):
            features.append(
                node_feats[
                    :,
                    i
                    * (self.l_max + 1) ** 2  # 1, 1 + 3, 1 + 3 + 5, ...
                    * self.num_features : ((i + 1) * (self.l_max + 1) ** 2)
                    * self.num_features,
                ]
            )
        features.append(node_feats[:, -self.num_features:])

        y = torch.cat([read(f) for read, f in zip(self.readouts, features)], dim=-1)
        for layer in self.mix:
            y = layer(y)

        # normalization breaks training!?
        #node_logits = y - torch.logsumexp(y, dim=-1, keepdim=True)
        #node_logits = y
        #node_deltas = y - torch.logsumexp(y, dim=-1, keepdim=True)
        node_deltas = y
        #node_deltas = (-1.0) * torch.nn.functional.log_softmax(y, dim=-1) # positive 
        #node_deltas = torch.nn.functional.softplus(y)
        #node_logits = self.scale * (-1.0) * (node_deltas + data['node_inter_es'].view(-1, 1))
        node_logits = self.scale * (-1.0) * node_deltas
        #node_logits = y - torch.logsumexp(y, dim=-1, keepdim=True)
        probabilities = torch.nn.functional.softmax(node_logits, dim=-1)
        #node_deltas = (-1.0) * node_logits
        node_delta = torch.sum(probabilities * node_deltas, dim=-1)
        node_inverse = torch.sum((1 - probabilities) * node_deltas, dim=-1)
        logits = scatter_sum(
                src=node_logits,
                index=data['batch'],
                dim=0,
                dim_size=data["ptr"].numel() - 1,
                )
        deltas = scatter_sum(
                src=node_deltas,
                index=data['batch'],
                dim=0,
                dim_size=data["ptr"].numel() - 1,
                )
        delta = scatter_sum(
                src=node_delta,
                index=data['batch'],
                dim=0,
                dim_size=data["ptr"].numel() - 1,
                )

        return {
                'node_logits': node_logits,
                'node_deltas': node_deltas,
                'node_delta': node_delta,
                'node_inverse': node_inverse,
                'logits': logits,
                'deltas': deltas,
                'delta': delta,
                }