###########################################################################################
# Atomic Data Class for handling molecules as graphs
# Authors: Sander Vandenhaute
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################


from typing import Optional, Callable
import functools

import numpy as np
import torch

from e3nn import o3, nn
from e3nn.util.jit import compile_mode

from mace.tools import TensorDict
from mace.tools.scatter import scatter_sum, scatter_mean


class ClassifierBlock(torch.nn.Module):

    def __init__(
            self,
            phases: list[str],
            num_interactions: int,
            hidden_irreps: o3.Irreps,
            layer_sizes: list[int],
            ):
        super().__init__()
        self.phases = phases
        self.l_max = max([ir[1][0] for ir in hidden_irreps])
        self.num_features = hidden_irreps[0].dim

        # Create indices to select scalar (l=0) features
        features_per_l = [(2 * l + 1) * self.num_features for l in range(self.l_max + 1)]
        total_features_per_block = sum(features_per_l)

        # Create selection indices
        indices = []
        curr_idx = 0
        for i in range(num_interactions - 1):
            # For each interaction block, select l=0 features
            indices.extend(range(curr_idx, curr_idx + features_per_l[0]))
            curr_idx += total_features_per_block

        # For final interaction, all features are l=0
        indices.extend(range(curr_idx, curr_idx + self.num_features))

        indices = torch.tensor(indices, dtype=torch.long)
        self.register_buffer('scalar_indices', indices)

        # Calculate total number of scalar features for MLP input
        num_scalar_features = len(indices)

        # Build MLP layers
        layers = []
        prev_size = num_scalar_features
        for size in layer_sizes:
            layers.append(torch.nn.Linear(prev_size, size))
            layers.append(torch.nn.SiLU())
            prev_size = size
        # Final layer to output phase logits
        layers.append(torch.nn.Linear(prev_size, len(phases)))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, Optional[torch.Tensor]]:
        node_feats = data['node_feats']

        # Select only scalar features using indices
        scalar_feats = node_feats[:, self.scalar_indices]

        # Process through MLP layers
        y = scalar_feats
        for layer in self.layers:
            y = layer(y)

        node_deltas = y
        node_logits = (-1.0) * node_deltas
        normalization = torch.logsumexp(node_logits, dim=-1, keepdim=True)
        probabilities = torch.nn.functional.softmax(node_logits - normalization, dim=-1)
        node_delta = torch.sum(probabilities * node_deltas, dim=-1)
        logits = scatter_sum(
            src=node_logits,
            index=data['batch'],
            dim=0,
            dim_size=data["ptr"].numel() - 1,
        )

        logits_forces: Optional[torch.Tensor] = None
        logits_stress: Optional[torch.Tensor] = None

        torch._C._debug_only_display_vmap_fallback_warnings(True)

        if not self.training:
            normalization = torch.logsumexp(logits, dim=1, keepdim=True)
            logits_n = logits - normalization
            forces_list = []
            for i in range(len(self.phases)):
                vector = torch.zeros_like(logits, device=logits.device)
                vector[:, i] = 1.0  # select phase i
                grads = torch.autograd.grad(
                    logits_n,
                    data['positions'],
                    grad_outputs=vector,
                    create_graph=False,
                    retain_graph=True,
                    allow_unused=False,
                    materialize_grads=False,
                )[0]

                forces_list.append((-1.0) * grads)
            logits_forces = torch.stack(forces_list, dim=0)

        return {
            'logits': logits,
            'logits_forces': logits_forces,
            'logits_stress': logits_stress,
            'node_delta': node_delta,
        }
