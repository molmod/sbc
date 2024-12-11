###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Sander Vandenhaute
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Any, Callable, Dict, List, Optional, Type
from functools import partial

import numpy as np
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from mace.modules import MACE
from mace.data import AtomicData
from mace.tools.scatter import scatter_sum, scatter_mean

from mace.modules.blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
    AgnesiTransform,
    SoftTransform,
    BesselBasis,
    GaussianBasis,
    ChebychevBasis
)
from mace.modules.utils import (
    compute_fixed_charge_dipole,
    compute_forces,
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
)

from sbc.modules.blocks import SimpleClassifier, EnergyBasedClassifier


def extract_kwargs(model):
    if hasattr(model.radial_embedding, 'distance_transform'):
        transform = model.radial_embedding.distance_transform
        if type(transform) is AgnesiTransform:
            distance_transform = 'Agnesi'
        elif type(transform) is SoftTransform:
            distance_transform = 'Soft'
        else:
            raise ValueError('unknown transform {}'.format(transform))
    else:
        distance_transform = 'None'

    bessel_fn = model.radial_embedding.bessel_fn
    if type(bessel_fn) is BesselBasis:
        radial_type = 'bessel'
    elif type(bessel_fn) is GaussianBasis:
        radial_type = 'gaussian'
    elif type(bessel_fn) is ChebychevBasis:
        radial_type = 'chebychev'
    else:
        raise ValueError('unknown basis {}'.format(bessel_fn))

    kwargs = {
            'r_max': model.radial_embedding.bessel_fn.r_max.item(),
            'num_bessel': torch.numel(model.radial_embedding.bessel_fn.bessel_weights),
            'num_polynomial_cutoff': model.radial_embedding.cutoff_fn.p.item(),
            'max_ell': model.spherical_harmonics.irreps_out.lmax,
            'interaction_cls': type(model.interactions[1]),
            'interaction_cls_first': type(model.interactions[0]),
            'num_interactions': model.num_interactions.item(),
            'num_elements': model.node_embedding.linear.irreps_in.dim,
            'hidden_irreps': model.readouts[0].linear.irreps_in,
            'MLP_irreps': model.readouts[-1].hidden_irreps,
            'atomic_energies': model.atomic_energies_fn.atomic_energies.cpu().numpy(),
            'avg_num_neighbors': model.interactions[0].avg_num_neighbors,
            'atomic_numbers': model.atomic_numbers.cpu().numpy(),
            'correlation': model.products[0].symmetric_contractions.contractions[0].correlation,
            'gate': torch.nn.functional.silu, # hardcoded
            'radial_MLP': model.interactions[0].radial_MLP,
            'radial_type': radial_type,
            'pair_repulsion': model.pair_repulsion,
            'distance_transform': distance_transform,
            'heads': None,
            }
    if hasattr(model, 'scale_shift'):
        kwargs['atomic_inter_scale'] = model.scale_shift.scale.item()
        kwargs['atomic_inter_shift'] = model.scale_shift.shift.item()
    if hasattr(model, 'classifier'):
        kwargs['phases'] = model.classifier.phases
        kwargs['classifier'] = model.classifier.__class__.__name__
        kwargs['classifier_readout'] = model.classifier.readout_layers
        kwargs['classifier_mixing'] = model.classifier.mixing_layer
    return kwargs


@compile_mode("script")
class ClassifierMACE(MACE):
    def __init__(
        self,
        num_interactions: int,
        hidden_irreps: o3.Irreps,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        phases: list[str],
        classifier: str,
        classifier_readout: list[int],
        classifier_mixing: Optional[int],
        **kwargs,
    ):
        super().__init__(
                **kwargs,
                num_interactions=num_interactions,
                hidden_irreps=hidden_irreps,
                )
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )
        if classifier == 'SimpleClassifier':
            classifier_cls = SimpleClassifier
        elif classifier == 'EnergyBasedClassifier':
            classifier_cls = EnergyBasedClassifier
        self.classifier = classifier_cls(
                phases=phases,
                num_interactions=num_interactions,
                hidden_irreps=hidden_irreps,
                readout_layers=classifier_readout,
                mixing_layer=classifier_mixing,
                )

    @classmethod
    def from_model(cls, model, **kwargs):
        model_kwargs = extract_kwargs(model)
        new_model = cls(**kwargs, **model_kwargs)
        new_model.load_state_dict( # loads existing model parameters
                model.state_dict(),
                strict=False,
                )
        return new_model

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_atoms_arange = torch.arange(data["positions"].shape[0])
        num_graphs = data["ptr"].numel() - 1
        node_heads = (
            data["head"][data["batch"]]
            if "head" in data
            else torch.zeros_like(data["batch"])
        )
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        )  # [n_graphs, num_heads]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
        else:
            pair_node_energy = torch.zeros_like(node_e0)
        # Interactions
        node_es_list = [pair_node_energy]
        node_feats_list = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
            )
            node_feats_list.append(node_feats)
            node_es_list.append(
                readout(node_feats, node_heads)[num_atoms_arange, node_heads]
            )  # {[n_nodes, ], }

        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)
        # Sum over interactions
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es, node_heads)

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Add E_0 and (scaled) interaction energy
        total_energy = e0 + inter_e
        node_energy = node_e0 + node_inter_es
        forces, virials, stress, hessian = get_outputs(
            energy=inter_e,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=True,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
        )

        cls_data = {
            'node_feats': node_feats_out,
            'batch': data['batch'],
            'ptr': data['ptr'],
            'positions': data['positions'],
        }
        out = self.classifier(cls_data)
        logits = out['logits']
        logits_forces = out['logits_forces']
        logits_stress = out['logits_stress']
        node_delta = out['node_delta']

        output = {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "hessian": hessian,
            "node_feats": node_feats_out,
            "logits": logits,
            "logits_forces": logits_forces,
            "logits_stress": logits_stress,
            "node_delta": node_delta,
        }

        return output


def hills(
    logits: np.ndarray,
    centers: np.ndarray,
    height: float,
    sigma: float,
) -> tuple[float, np.ndarray]:
    assert len(logits) == centers.shape[1]
    logits = logits.reshape((1, -1))

    delta = logits - centers
    exponent = (-1.0) * np.sum(delta ** 2, axis=1) / (2 * sigma ** 2)
    energy = np.sum(height * np.exp(exponent))

    extra = (-1.0) * delta / (2 * sigma ** 2)
    gradient = np.sum(height * extra * np.exp(exponent), axis=0)
    return energy, gradient
